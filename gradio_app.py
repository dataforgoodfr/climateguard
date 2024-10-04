import os
from dotenv import load_dotenv
import gradio as gr
import pandas as pd
from anthropic import Anthropic
from pydantic import BaseModel

from climateguard.models import Claims

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def read_parquet(uploaded_file):
    return pd.read_parquet(uploaded_file)

# Initialize the Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY)


# ... (keep the detect_claims, analyze_disinformation, and read_parquet functions as they are) ...

MAPPING = {
    "very low": 0,
    "low": 1,
    "medium": 2,
    "high": 5,
}

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

    completion = client.completions.create(
        model="claude-2.1",
        prompt=prompt,
        max_tokens_to_sample=2000,
        temperature=0.2,
    )
    response_content = completion.completion
    
    # Parse the JSON response
    claims_data = Claims.model_validate_json(response_content)
    result = pd.DataFrame([claim.dict() for claim in claims_data.claims])

    return result, completion.usage.total_tokens


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



def detect_claims_interface(text_input, source_type):
    claims, _ = detect_claims(text_input)
    claims["score"] = claims["disinformation_score"].map(lambda x: MAPPING.get(x))
    average_score = round(claims["score"].mean(), 1)
    
    output = f"Disinformation risk average score: {average_score}/5\n\n"
    
    for _, row in claims.iterrows():
        score = row["score"]
        risk_level = "Low" if score <= 1 else "Medium" if score == 2 else "High"
        
        output += f"Claim: {row['claim']}\n"
        output += f"Context: {row['context']}\n"
        output += f"Disinformation risk score: {row['disinformation_score']} ({row['score']}/5)\n"
        output += f"Analysis: {row['analysis']}\n\n"
    
    return output, claims

def analyze_disinformation_interface(claim, sources):
    analysis, _ = analyze_disinformation(claim)
    return analysis

def generate_alert(alert_type):
    # Placeholder function for alert generation
    return f"Generated {alert_type} alert"

def gradio_app():
    with gr.Blocks() as app:
        gr.Markdown("# CLIMATE DISINFORMATION DETECTION")
        
        with gr.Tab("Climate speech detection"):
            file_upload = gr.File(label="Choose a file to analyze", file_types=[".parquet"])
            output_dataframe = gr.Dataframe()
            
            file_upload.upload(read_parquet, file_upload, output_dataframe)
        
        with gr.Tab("Claims detection"):
            text_input = gr.Textbox(label="Enter text for analysis", lines=5)
            source_type = gr.Dropdown(["TV / Radio", "Social network post", "Video transcript"], label="Select source type")
            detect_button = gr.Button("Analyze")
            claims_output = gr.Textbox(label="Analysis Results", lines=10)
            
            detect_button.click(detect_claims_interface, inputs=[text_input, source_type], outputs=[claims_output])
        
        with gr.Tab("Climate disinformation analysis"):
            claim_dropdown = gr.Dropdown(label="Select claim")
            sources = gr.CheckboxGroup(["IPCC", "IPBES", "ADEME", "ClimateFeedback"], label="Select sources", value=["IPCC", "IPBES", "ADEME", "ClimateFeedback"])
            analysis_output = gr.Textbox(label="Analysis", lines=10)
            
            claim_dropdown.change(analyze_disinformation_interface, inputs=[claim_dropdown, sources], outputs=analysis_output)
        
        with gr.Tab("Alert generation"):
            alert_type = gr.Radio(["ARCOM", "DSA"], label="Select alert type")
            generate_button = gr.Button("Generate Alert")
            alert_output = gr.Textbox(label="Alert Output", lines=5)
            
            generate_button.click(generate_alert, inputs=alert_type, outputs=alert_output)
    
    return app

if __name__ == "__main__":
    app = gradio_app()
    app.launch()