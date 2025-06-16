import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import openai
from llama_index.core.node_parser import SentenceSplitter
from prompts import DisinformationPrompt
from secret_utils import get_secret_docker
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import nn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
    # Keywords ID
    id: Optional[str] = None


@dataclass
class PipelineOutput:
    # disinformation score (0: not disinformation - 10: disinformation)
    score: int
    # a brief reason about the score
    reason: str = ""
    # suggestion of other metadata that could be added
    cards_category: Optional[str] = None
    # probability of generation
    probability: Optional[float] = None
    # Keywords ID
    id: Optional[str] = None


class Pipeline(ABC):
    @abstractmethod
    def process(self, input_data: PipelineInput) -> PipelineOutput:
        """Process input data and return an integer classification."""
        pass

    @abstractmethod
    def batch_process(self, input_data: List[PipelineInput]) -> List[PipelineOutput]:
        """Process input data and return an integer classification for each input in batch."""
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
    def __init__(
        self, model_name: str, api_key: str, prompt: DisinformationPrompt
    ) -> None:
        openai_key = get_secret_docker("OPENAI_API_KEY")
        openai.api_key = openai_key
        os.environ["OPENAI_API_KEY"] = openai_key
        self._model = model_name

        self._system_prompt = prompt.prompt
        self.prompt_version = prompt.version
        self.version = f"{model_name}/{prompt.version}"
        self._steps = [
            f"Single Open AI prompt with {self._model} - prompt version: {prompt.version} - prompt text: {self._system_prompt}"
        ]

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
        
    def batch_process(self, input_data: List[PipelineInput]):
        responses = []
        for data in input_data:
            responses.append(
                self.process(data)
            )
        return responses

    def describe(self) -> None:
        for step in self._steps:
            logging.info(step)
        return self._steps


class BertPipeline(Pipeline):
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str = None,
        chunk_size: int = 512,
        chunk_overlap: int = 256,
        batch_size: int = 32,
        verbose: bool = False,
    ):
        self._model = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        if tokenizer_name is None:
            tokenizer_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.version = f"{model_name}/{chunk_size}_{chunk_overlap}"
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.verbose = verbose

        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self._steps = [
            (
                f"BERT Pipeline using model: {self._model} - ",
                f"chunk size: {chunk_size} - prompt text: {chunk_overlap}",
            )
        ]

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

    def batch_process(self, input_data: List[PipelineInput]):
        ids = []
        texts = []
        for idx, data_record in enumerate(input_data):
            chunks = self.splitter.split_text(data_record.transcript)
            for chunk in chunks:
                _id = data_record.id if data_record.id else idx
                ids.append(_id)
                texts.append(chunk)
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.chunk_size,
            return_tensors="pt",
        )
        predictions = []
        probabilities = []
        logging.info(
            f"Processing {len(inputs['input_ids'])} texts "
            f"with batch size {self.batch_size}: {len(inputs["input_ids"]) // self.batch_size + 1} iterations"
        )
        for window in range(len(inputs["input_ids"]) // self.batch_size + 1):
            if self.verbose:
                logging.info(f"Processing Batch number : {window+1}")
            outputs = self.model(
                input_ids=inputs["input_ids"][self.batch_size * window: self.batch_size * (window+1)],
                attention_mask=inputs["attention_mask"][self.batch_size * window: self.batch_size * (window+1)],
                seq_len=self.chunk_size,
                batch_size=self.batch_size,
                show_progress=True,
            )
            predictions.extend(outputs.logits.numpy().argmax(1).tolist())
            probabilities.extend(nn.functional.softmax(outputs.logits, dim=-1).numpy().max(1).tolist())
        logging.info(len(predictions))
        results_df = pd.DataFrame(
            {
                "id": ids,
                "prediction": predictions,
                "probability": probabilities,
            }
        )
        results_df = results_df.groupby(["id"]).agg("max").reset_index()
        logging.info(results_df.info())
        return [
            PipelineOutput(id=row["id"], score=row["prediction"], probability=row["probability"])
            for idx, row in results_df.iterrows()
        ]
