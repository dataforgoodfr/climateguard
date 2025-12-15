import requests
import pandas as pd
import math
from datasets import load_dataset, concatenate_datasets, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login
import os
import argparse
import time
from tqdm import tqdm
from typing import Optional, Union
import re
from sklearn.metrics import classification_report 

from openai import OpenAI


load_dotenv()
login(token=os.getenv("HF_TOKEN"))

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv("VLLM_APIKEY"),
    base_url=os.getenv("VLLM_HOST"),
)

prompt = """You are an assistant helping editors to moderate TV and radio content.
You will be provided with a transcript delimited by triple backticks.
Bare in mind that the transcript may be missing punctuation and may be of very low quality, with incorrect vocabulary, cuts in the wrong places, or may include some phonetic transcription.

Does the text promote climate change misinformation that undermines well-established scientific consensus, such as denying the existence of climate change or the factors that contribute to it ?

Answer just a number between 0 - does not promote - and 1 - certainly promotes.

text: {transcript}"""


def get_data(args):
    dataset = load_dataset("DataForGood/climateguard")

    dataset = dataset.filter(lambda example: example["comments"] == [])
    # dataset = dataset.filter(lambda example: example["year"] == 2025)
    dataset = dataset.filter(
        lambda example: isinstance(example["misinformation"], bool)
    )
    dataset = dataset.filter(
        lambda example: not isinstance(example["misinformation"], str)
    )
    dataset = dataset.map(
        lambda example: {
            "text": example["plaintext"],
            "value": int(example["misinformation"]),
        }
    )
    if args.test_split == "time":
        concat_dataset = concatenate_datasets([dataset["train"], dataset["test"]])

        concat_dataset = concat_dataset.sort(["year", "month", "day"])
        stop_index = math.floor(len(concat_dataset) * 0.80)
        stop_date = pd.to_datetime(
            concat_dataset.select([stop_index])["start"][0]
        ).date()
        print(f"Setting train data cutoff at {stop_date.strftime('%Y-%m-%d')}")
        _train_data = concat_dataset.filter(
            lambda example: pd.to_datetime(example["start"]).date() <= stop_date
        )
        _test_data = concat_dataset.filter(
            lambda example: pd.to_datetime(example["start"]).date() > stop_date
        )
        dataset = DatasetDict(
            {
                "train": _train_data,
                "test": _test_data,
            }
        )
    dataset = dataset.select_columns(["id", "text", "value"])
    # Print dataset stats:
    print("Train dataset example split:")
    print(dataset["train"].to_pandas()["value"].value_counts())
    print("Test dataset example split:")
    print(dataset["test"].to_pandas()["value"].value_counts())
    return dataset


def get(client, prompt):

    models = client.models.list()
    model = models.data[0].id

    messages = [
        {"role": "user", "content": prompt}
    ]
    # Chat Completion API
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.0,
        max_tokens=10,
    )
    print(chat_completion)
    return chat_completion


def parse_response(response: Optional[Union[int, str]]):
    """Parse response containing only a score."""
    if isinstance(response, int):
        return response
    match = re.match(r"^[^\d]*(\d+)", response)
    if match:
        score = int(match.group(1))  # Extract score as an integer
    else:
        print("Could not parse response")
        score = 0
    return score


def test(client, test_dataset):
    results = []
    raw_results = []
    dataset = dataset.map(
        lambda example: {
            "messages": [
                {"role": "user", "content": prompt.format(transcript=example["text"])},
                {"role": "assistant", "content": str(example["value"])},
            ]
        }
    )
    t=time()
    for example in tqdm(test_dataset):
        prompt = prompt.format(transcript=example["text"])
        prediction = get(client=client, prompt=prompt)
        raw_results.append(prediction)
        results.append(int(parse_response(prediction)))

    t_taken = time() - t
    print(f"Time taken: {t_taken}; Time per sample: {t_taken / len(test_dataset): .2f}")
    report = classification_report(
        test_dataset.to_pandas()["value"].astype(int),
        results,
    )
    print(f"Classification on test set: \n{report}")
    df_results = pd.DataFrame(
        {
            "predictions": results,
            "labels": test_dataset.to_pandas()["value"].astype(int),
            "responses": raw_results,
        }
    )
    print(df_results.head())

    # df_results.to_csv("vllm_tests.csv", index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-split", type=str, default="default")
    
    args = parser.parse_args()
    print(args)
    dataset = get_data()
    test_dataset = dataset["test"]
    for example in test_dataset:
        response = get(client=client, prompt=prompt.format(transcript=example["text"]))
        print(response)
        break