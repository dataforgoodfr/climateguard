import streamlit as st
import pandas as pd
import os

# Set page configuration
st.set_page_config(page_title="CLIMATE DISINFORMATION", layout="wide")

# Set font to Poppins
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');
    html, body, h1, h2, h3, h4, h5, h6, p, div, span, button, label, input, select, textarea {
        font-family: 'Poppins', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main app
st.title("CLIMATE DISINFORMATION DETECTION")


from openai import OpenAI
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

from pydantic import BaseModel
class Claim(BaseModel):
    claim: str
    context: str
    analysis: str
    disinformation_score: str
    disinformation_category: str

class Claims(BaseModel):
    claims: list[Claim]

@st.cache_data
def detect_claims(transcription):
    prompt = f"""

Tu es expert en désinformation sur les sujets environnementaux, expert en science climatique et sachant tout sur le GIEC. Je vais te donner un extrait d'une retranscription de 2 minutes d'un flux TV ou Radio. 
A partir de cet extrait liste moi tous les faits/opinions environnementaux (claim) uniques qu'il faudrait factchecker. Et pour chaque claim, donne une première analyse si c'est de la désinformation ou non, un score si c'est de la désinformation, ainsi qu'une catégorisation de cette allégation.
Ne sélectionne que les claims sur les thématiques environnementales (changement climatique, transition écologique, énergie, biodiversité, pollution, pesticides, ressources (eau, minéraux, ..) et pas sur les thématiques sociales et/ou économiques
Renvoie le résultat en json sans autre phrase d'introduction ou de conclusion avec à chaque fois les 5 champs suivants : "claim",analysis","disinformation_score","disinformation_category"

- "claim" - l'allégation à potentiellement vérifier
- "context" - reformulation du contexte dans laquelle cette allégation a été prononcée (maximum 1 paragraphe)
- "analysis" - première analyse du point de vue de l'expert sur le potentiel de désinformation de cette allégation en fonction du contexte

Pour les scores "disinformation_score"
- "very low" = pas de problème, l'allégation n'est pas trompeuse ou à risque. pas besoin d'investiguer plus loin
- "low" = allégation qui nécessiterait une vérification et une interrogation, mais sur un sujet peu important et significatif dans le contexte des enjeux écologiques (exemple : les tondeuses à gazon, 
- "medium" = allégation problématique sur un sujet écologique important (scientifique, impacts, élections, politique, transport, agriculture, énergie, alimentation, démocratie ...) , qui nécessiterait vraiment d'être vérifiée, déconstruite, débunkée et interrogée. En particulier pour les opinions fallacieuses
- "high" = allégation grave, en particulier si elle nie le consensus scientifique

Pour les catégories de désinformation "disinformation_category": 
- "consensus" = négation du consensus scientifique
- "facts" = fait à vérifier, à préciser ou contextualiser
- "narrative" = narratif fallacieux ou opinion qui sème le doute (par exemple : "les écolos veulent nous enlever nos libertés")
- "other"

<transcription>
{transcription}
</transcription>
    """

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        response_format=Claims,
    )
    n_tokens = completion.usage.total_tokens
    claims_data = completion.choices[0].message.parsed
    result = pd.DataFrame([claim.dict() for claim in claims_data.claims])

    return result,n_tokens


@st.cache_data
def analyze_disinformation(claim):

    prompt = f"""
Tu es expert du changement climatique, scientifique du GIEC. 
Voici une allégation qui pourrait s'apparenter à de la désinformation sur les enjeux écologiques prononcées à la TV. 
{claim}
Peux-tu en faire une analyse complète de pourquoi c'est de la désinformation, puis en débunkant de façon sourcée. 
Renvoie directement ton analyse sans message d'introduction ou de conclusion.
    """

    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        # response_format=Claims,
    )
    n_tokens = completion.usage.total_tokens
    message = completion.choices[0].message.content

    return message,n_tokens

@st.cache_data
def read_parquet(uploaded_file):
    return pd.read_parquet(uploaded_file)


MAPPING = {
    "very low":0,
    "low":1,
    "medium":2,
    "high":5,
}




# Create tabs
tab0,tab1, tab2, tab3 = st.tabs(["Climate speech detection ","Claims detection", "Climate disinformation analysis", "Alert generation"])


with tab0:

    import pandas as pd

    # File uploader
    uploaded_file = st.file_uploader("Choose a file to analyze", type="parquet")

    if uploaded_file is not None:
        # Read the parquet file
        df_claims = read_parquet(uploaded_file)
        
        # Display the dataframe
        st.dataframe(df_claims)


# Tab 1: Claims Detection
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        text_input = st.text_area("Enter text for analysis", height=200)

    with col2:
        source_type = st.selectbox("Select source type", ["TV / Radio", "Social network post", "Video transcript"])
        detect_button = st.button("Analyze", key="analyze_claims")

    if detect_button:
        claims,_ = detect_claims(text_input)
        claims["score"] = claims["disinformation_score"].map(lambda x : MAPPING.get(x))
        average_score = round(claims["score"].mean(),1)

        st.metric("Disinformation risk average score",f"{average_score}/5")

        for i,row in claims.iterrows():

            score = row["score"]

            col1, col2 = st.columns([1, 1])
            with col1:
                if score == 0:
                    st.success(f"### {row['claim']}\n{row['context']}")
                elif score == 1:
                    st.info(f"### {row['claim']}\n{row['context']}")
                elif score == 2:
                    st.warning(f"### {row['claim']}\n{row['context']}")
                else:
                    st.error(f"### {row['claim']}\n{row['context']}")

            with col2:
                st.metric("Disinformation risk score",f"{row['disinformation_score']} ({row['score']}/5)")
                st.markdown(f"{row['analysis']}")



    else:
        claims = pd.DataFrame()


# Tab 2: Climate Disinformation Analysis
with tab2:
    col1, col2 = st.columns([2, 1])

    if len(claims)> 0:

        options = claims["claim"].tolist()


        with col1:
            dropdown_selection = st.selectbox("Select claim", options)
            claim_selected = claims[claims["claim"]==dropdown_selection].iloc[0]


        with col2:
            sources = st.multiselect("Select sources", ["IPCC", "IPBES", "ADEME", "ClimateFeedback"],default = ["IPCC", "IPBES", "ADEME", "ClimateFeedback"])

        analysis,_ = analyze_disinformation(dropdown_selection)
        st.markdown(analysis)
    

# Tab 3: Alert Generation
with tab3:
    alert_type = st.selectbox("Select alert type", ["ARCOM", "DSA"])
    generate_button = st.button("Generate Alert")
    st.text_area("Alert Output", height=200, key="alert_output")

