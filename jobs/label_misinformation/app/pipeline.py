from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import List, Optional
from secret_utils import get_secret_docker
import os
import openai
import re
from secret_utils import *

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
    # a brief reason about the score
    reason: str = ""
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


def parse_response_reason(response: str) -> PipelineOutput:
    """Parse llm output containing a score and a reason."""
    # "Score: 0, Reason: score too low"
    match = re.match(r"Score: *(\d+), *Reason: *(.+)", response)
    if match:
        score = int(match.group(1))  # Extract score as an integer
        reason = match.group(2)  # Extract reason
    else:
        logging.warning(f"Could not parse {response}")
        score = 0
        reason = "too low"
    logging.info(f"Parsed score: {score}, reason: {reason}")
    return PipelineOutput(score=score, reason=reason)


def parse_response(response: str) -> PipelineOutput:
    """Parse response containing only a score."""
    match = re.match(r"^[^\d]*(\d+)", response)
    if match:
        score = int(match.group(1))  # Extract score as an integer
    else:
        logging.warning(f"Could not parse {response}")
        score = 0
    logging.info(f"Parsed score: {score}")
    return PipelineOutput(score=score)


class SinglePromptPipeline(Pipeline):
    def __init__(self, model_name: str, api_key: str) -> None:
        openai_key = get_secret_docker("OPENAI_API_KEY")
        openai.api_key = openai_key
        os.environ["OPENAI_API_KEY"] = openai_key
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
            openai_key = get_secret_docker("OPENAI_API_KEY")
            openai.api_key = openai_key
            response = openai.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0,
            )
            logging.debug(f"Response API: {response}")
            result = response.choices[0].message.content.strip()

            return parse_response(result)
        except Exception as e:
            logging.error(f"Error : {e}")
            raise Exception

    def describe(self) -> None:
        for step in self._steps:
            logging.info(step)
        return self._steps
