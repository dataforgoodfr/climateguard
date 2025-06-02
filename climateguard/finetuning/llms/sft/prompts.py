prompt_chat = """You are an assistant helping editors to moderate TV and radio content.
 You will be provided with a transcript delimited by triple backticks.
 Bear in mind that the transcript may be missing punctuation and may be of very low quality, with incorrect vocabulary, cuts in the wrong places, or may include some phonetic transcription.

 Does the text promote climate change misinformation that undermines well-established scientific consensus, such as denying the existence of climate change or the factors that contribute to it ?
 
 Answer in two parts:
 1 An analysis of the extract and whether it contains or promotes climate change misinformation.
 2. A number on a scale of 0 - does not promote - to 10 - certainly promotes (the score).

 Format your response always like this:
 \"{reason: X, score: Y}\"
 Always starting with the reason.
 Here is the text:
 """

prompt_synthetic_misinformation = (
    "You are a synthetic datset generator. I have a piece of text that has been classified as "
    "disinformation on climate change and the environment. The text will be a transcript from a"
    "french tv or radio program\n"
    "Instructions:\n"
    "- Analyze the text and return an explaination as to why this consititutes climate change disinformation\n"
    "- Keep your response to one or two sentences.\n"
    "- Do not return any other content appart from the explaination.\n\n"
    "Here is the text:\n"
)

prompt_synthetic_information = (
    "You are a synthetic datset generator. I have a piece of text that has been classified as "
    "disinformation on climate change and the environment. The text will be a transcript from a"
    "french tv or radio program\n"
    "Instructions:\n"
    "- Analyze the text and return an explaination as to why this does NOT consititute climate change disinformation\n"
    "- Keep your response to one or two sentences.\n"
    "- Do not return any other content appart from the explaination.\n\n"
    "Here is the text:\n"
)