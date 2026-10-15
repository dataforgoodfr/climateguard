from dataclasses import dataclass


@dataclass
class DisinformationPrompt:
    prompt: str
    version: str
    prod: bool = False
    # True: the model outputs a binary misinformation flag (score in {0, 10}) rather than a
    # continuous 0-10 scale. MIN_MISINFORMATION_SCORE is ignored for binary prompts; score > 0 is
    # used instead (see detect_misinformation in main.py).
    binary: bool = False

    def __str__(self):
        return f"DisinformationPrompt Version: {self.version} | Production Use: {self.prod} | Prompt Text: {self.prompt}"


PROMPTS = {
    "0.1.0": DisinformationPrompt(
        prompt="""
Tu es un assistant qui aide des éditeurs à modérer des contenus TV et radio.
Tu vas recevoir une transcription délimitée par des triples guillemets.
Attention, cette transcription peut être mal ponctuée et de très mauvaise qualité, avec un vocabulaire incorrect, des coupures mal placées, ou une transcription phonétique approximative.

La désinformation climatique est tout contenu qui contredit le consensus scientifique établi ou propage des narratifs trompeurs sur le changement climatique, selon trois axes :
- Science climatique : nier ou minimiser les causes humaines du réchauffement, contester l'existence ou la gravité de la crise climatique, ou déformer les projections du GIEC.
- Action climatique : discréditer les politiques climatiques (Accord de Paris, lois climat), présenter l'inaction comme légitime, ou instrumentaliser de fausses données pour bloquer la régulation.
- Solutions d'atténuation et d'adaptation : tromper sur l'efficacité, le coût ou la faisabilité des solutions reconnues par le GIEC (renouvelables, efficacité énergétique, capture de carbone, etc.).

Sont également concernés les chiffres falsifiés ou sortis de contexte, les corrélations abusives, les théories du complot sur les acteurs de la transition, et les amalgames entre action climatique et agendas idéologiques sans base factuelle.

Le texte ci-dessous promeut-il de la désinformation climatique telle que définie ci-dessus ?
En cas de doute, considère que le texte promeut de la désinformation : nous cherchons à maximiser le rappel, il vaut mieux un faux positif qu'un faux négatif.

Réfléchis d'abord brièvement (2 à 3 phrases maximum) en donnant seulement ta propre analyse, sans jamais citer ni reformuler entre guillemets un extrait du texte, puis donne ta réponse finale.
N'utilise aucun markdown (pas de gras, italique, listes, titres ou guillemets typographiques) : réponds uniquement en texte brut.
N'ouvre jamais de guillemets ("), ils provoquent des boucles infinies de génération.

texte: ```{transcript}```
""",
        version="0.1.0",
        prod=True,
        binary=True,
    ),
    "0.0.1": DisinformationPrompt(
        prompt="""You are an assistant helping editors to moderate TV and radio content.
You will be provided with a transcript delimited by triple backticks.
Bear in mind that the transcript may be missing punctuation and may be of very low quality, with incorrect vocabulary, cuts in the wrong places, or may include some phonetic transcription.

Does the text promote climate change misinformation that undermines well-established scientific consensus, such as denying the existence of climate change or the factors that contribute to it ?

Answer just a number on a scale of 0 - does not promote - to 10 - certainly promotes.

text:""",
        version="0.0.1",
        prod=True,
    ),
    "0.0.0": DisinformationPrompt(
        prompt="""
You are an assistant helping editors to moderate TV and radio content.
You will be provided with a transcript delimited by triple backticks.
Bear in mind that the transcript may be missing punctuation and may be of very low quality, with incorrect vocabulary, cuts in the wrong places, or may include some phonetic transcription.
Does the text promote climate change misinformation that undermines well-established scientific consensus, such as denying the existence of climate change or the factors that contribute to it ?
Answer in two parts:
1. A number on a scale of 0 - does not promote - to 10 - certainly promotes (the score).
2. If the score is greater than 8, provide a brief explanation of why you gave this score (the reason). If the score is 8 or lower, provide the string 'score too low'
Format your response always like this:
"Score: X, Reason: Y"
Where X is the score (default if empty value to 0) and Y is the reason (default if empty is 'too low').
text:""",
        version="0.0.0",
        prod=False,
    ),
}
