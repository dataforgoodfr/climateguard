from abc import ABC, abstractmethod


class Prompt(ABC):
    @abstractmethod
    def __init__(self, **kwargs) -> str:
        """
        Sets the system prompt with keyword, prompt based parameters
        Returns the str of completed prompt.
        Ex: allows to inject few shot examples, metadata, etc.
            depending on the prompt's own definition.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, user_entry: str) -> str:
        """Returns the prompt completed with the user entry."""
        raise NotImplementedError

    @abstractmethod
    def get_system_prompt(self) -> str:
        """In case the prompt needs to be divided for the client."""
        raise NotImplementedError


class BasicPrompt(Prompt):
    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt

    def __call__(self, user_entry: str) -> str:
        return self._system_prompt + user_entry

    def get_system_prompt(self) -> str:
        return self._system_prompt


# this should probably not be in domain, but idealy would be accessible
# for usecases such as benchmarking etc.. ?
BASIC_PROMPT = BasicPrompt(
    system_prompt="""
        You are an assistant helping editors to moderate TV and radio content.
        You will be provided with a transcript delimited by triple backticks.
        Bear in mind that the transcript may be missing punctuation and may be of very low quality, with incorrect vocabulary, cuts in the wrong places, or may include some phonetic transcription.

        Does the text promote climate change misinformation that undermines well-established scientific consensus, such as denying the existence of climate change or the factors that contribute to it ?

        Answer just a number on a scale of 0 - does not promote - to 10 - certainly promotes. 


        text: """,
)
