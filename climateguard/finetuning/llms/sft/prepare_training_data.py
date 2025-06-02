import argparse
import json
import os
from typing import Any, Dict, Union

from numpy.random import randint
import jsonlines
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from prompts import (
    prompt_chat,
    prompt_synthetic_information,
    prompt_synthetic_misinformation,
)
from tqdm import tqdm

load_dotenv()
login(token=os.getenv("HF_TOKEN"))


def synth_gen_model_reason(
    client: OpenAI, record: Dict[str, Any], model: str = "gpt-4o-mini"
):
    prompt_synthetic = (
        prompt_synthetic_misinformation
        if record["misinformation"]
        else prompt_synthetic_information
    )
    response = client.responses.create(
        model=model, input=prompt_synthetic + record["plaintext"]
    )
    return response.output[0].content[0].text


def generate_conversations(client: OpenAI, dataset: Union[DatasetDict, Dataset]):
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]
    conversations = []
    for record in tqdm(dataset):
        messages = [{"role": "user", "content": prompt_chat + record["plaintext"]}]
        if len(record["model_reason"]) == 0:
            record["model_reason"] = synth_gen_model_reason(
                client=client, record=record
            )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "reason": record["model_reason"],
                        "score": int(record["misinformation"]) * 10,
                    }
                ),
            }
        )
        conversations.append({"messages": messages})
    return conversations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="DataForGood/climateguard")
    parser.add_argument("-t", "--train", type=str, default="data/train.jsonl")
    parser.add_argument("-t", "--val", type=str, default="data/valid.jsonl")
    parser.add_argument("-s", "--split", type=float, default=0.1)
    args = parser.parse_args()

    train_dataset = load_dataset(
       args.dataset, download_mode="force_redownload", split="train"
    )

    client = OpenAI()
    conversations = generate_conversations(client=client, dataset=train_dataset)

    n_validation = int((len(conversations) * args.split) // 1)
    validation_idxs = randint(0, len(conversations), n_validation)
    val_conversations = [conversations[idx] for idx in validation_idxs]
    train_conversations = [conversations[idx] for idx in range(len(conversations)) if idx not in validation_idxs]

    with jsonlines.open(args.train, "w") as writer:
        writer.write_all(train_conversations)

    with jsonlines.open(args.val, "w") as writer:
        writer.write_all(train_conversations)
