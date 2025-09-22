import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def live_detect():
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

        return result, n_tokens

    MAPPING = {
        "very low": 0,
        "low": 1,
        "medium": 2,
        "high": 5,
    }

    col1, col2 = st.columns([2, 1])

    with col1:
        text_input = st.text_area("Enter text for analysis", height=200)

    with col2:
        source_type = st.selectbox(
            "Select source type",
            ["TV / Radio", "Social network post", "Video transcript"],
        )
        detect_button = st.button("Analyze", key="analyze_claims")

    if detect_button:
        claims, _ = detect_claims(text_input)
        claims["score"] = claims["disinformation_score"].map(lambda x: MAPPING.get(x))
        average_score = round(claims["score"].mean(), 1)

        st.metric("Disinformation risk average score", f"{average_score}/5")

        for i, row in claims.iterrows():
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
                st.metric(
                    "Disinformation risk score",
                    f"{row['disinformation_score']} ({row['score']}/5)",
                )
                st.markdown(f"{row['analysis']}")

    else:
        claims = pd.DataFrame()
