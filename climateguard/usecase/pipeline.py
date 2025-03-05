from abc import ABC, abstractmethod
import logging
import re
from typing import List, Tuple
import openai
from climateguard.domain.documents import (
    ClassifiedDocument,
    DisinformationGradeEnum,
    Document,
)
from climateguard.domain.prompts import BASIC_PROMPT, Prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Pipeline(ABC):
    @abstractmethod
    def process(self, documents: List[Document]) -> List[ClassifiedDocument]:
        """Run a set of documents through the pipeline, and complete them."""
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> List[str]:
        """Describe pipeline steps - log / return"""
        raise NotImplementedError


class ModelClient(ABC):
    """
    A client configuration for a given model.
    Each ModelClient is defined for a single model call to the client.
    """

    @abstractmethod
    def call(self, prompt: Prompt, user_text: str) -> str:
        """
        Execute the call for a given prompt.
        NOTE: prompt and user_text are important in case the client separates them
        when calling the api.
        """
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> str:
        """Returns a description of the client and model."""
        raise NotImplementedError


class OpenAIClient(ModelClient):
    def __init__(self, api_key: str, model_name: str) -> None:
        openai.api_key = api_key
        self._model = model_name

    def call(self, prompt: Prompt, user_entry: str) -> str:
        messages = [{"role": "user", "content": prompt(user_entry)}]
        try:
            response = openai.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0,
            )
            logging.debug(f"Response API: {response}")
            result = response.choices[0].message.content.strip()
            return result

        except Exception as e:
            logging.error(f"Error : {e}")
            raise Exception


def parse_basic_response(response: str) -> Tuple[int, str]:
    """Parse a prompt from a basic prompt expecting a score and a reason."""
    # must parse  "Score: 0, Reason: score too low"
    match = re.match(r"Score: (\d+), Reason: (.+)", response)
    if match:
        score = int(match.group(1))  # Extract score as an integer
        reason = match.group(2)  # Extract reason
    else:
        logging.warning(f"Could not parse {response}")
        score = 0
        reason = "too low"
    logging.info(f"Parsed score: {score}, reason: {reason}")
    return score, reason


class SinglePromptPipeline(Pipeline):
    def __init__(self, model_client: ModelClient) -> None:
        self._prompt = BASIC_PROMPT
        self._steps = [
            f"Single Open AI prompt with {model_client.describe()} - prompt: {self._prompt(user_prompt='')}"
        ]
        self._model_client = model_client

    def process(self, documents: List[Document]) -> List[ClassifiedDocument]:
        results: List[ClassifiedDocument] = []
        for document in documents:
            response = self._model_client.call(prompt=self._prompt, user_text=document.transcript)
            score, reason = parse_basic_response(response)
            result = ClassifiedDocument(
                document,
                disinformation_document=(score == 10),
                disinformation_label=DisinformationGradeEnum(score),
                reason=reason,
            )
            results.append(result)

        return results

    def describe(self) -> None:
        for step in self._steps:
            logging.info(step)
        return self._steps
