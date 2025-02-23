from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import List, Optional

import openai

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class PipelineInput:
    transcript: str

    # suggestion of other metadata that could be added
    # path to audio file for whisperization...
    audio_file: Optional[str] = None
    # source of the transcript ex: "sud-radio"
    source: Optional[str] = None
    # date and time of the original emission
    date: Optional[str] = None


@dataclass
class PipelineOutput:
    # disinformation score (0: not disinformation - 10: disinformation)
    score: int
    # suggestion of other metadata that could be added
    cards_category: Optional[str] = None


class Pipeline(ABC):
    @abstractmethod
    def process(self, input_data: PipelineInput) -> PipelineOutput:
        """Process input data and return an integer classification."""
        pass

    @abstractmethod
    def describe(self) -> List[str]:
        """Describe pipeline steps - log / return"""
        pass


class SinglePromptPipeline(Pipeline):
    def __init__(self, model_name: str, api_key: str) -> None:
        openai.api_key = api_key
        self._model = model_name

        self._system_prompt = """
    You are an assistant helping editors to moderate TV and radio content.
    You will be provided with a transcript delimited by triple backticks.
    Bear in mind that the transcript may be missing punctuation and may be of very low quality, with incorrect vocabulary, cuts in the wrong places, or may include some phonetic transcription.

    Does the text promote climate change misinformation that undermines well-established scientific consensus, such as denying the existence of climate change or the factors that contribute to it ?

    Answer just a number on a scale of 0 - does not promote - to 10 - certainly promotes. 


    text:"""
        self._steps = [f"Single Open AI prompt with {self._model} - prompt: {self._system_prompt}"]

    def process(self, input_data: PipelineInput) -> int:
        prompt = self._system_prompt + f" '''{input_data.transcript}'''"
        messages = [{"role": "user", "content": prompt}]
        logging.debug(f"Send {messages}")

        try:
            response = openai.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0,
            )
            logging.info(f"Response API: {response} for text {input_data.transcript}")

            return int(response)
        except Exception as e:
            logging.error(f"Error : {e}")
            raise Exception

    def describe(self) -> None:
        for step in self._steps:
            logging.info(step)
        return self._steps
