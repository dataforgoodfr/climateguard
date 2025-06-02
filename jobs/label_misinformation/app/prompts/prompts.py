from dataclasses import dataclass


@dataclass
class DisinformationPrompt:
    prompt: str
    version: str
    prod: bool = False

    def __str__(self):
        return f"DisinformationPrompt Version: {self.version} | Production Use: {self.prod} | Prompt Text: {self.prompt}"


PROMPTS = {
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
