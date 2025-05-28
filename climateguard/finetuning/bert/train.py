import random
from datetime import datetime

import numpy as np
import pandas as pd
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from dotenv import load_dotenv
from huggingface_hub import HfFolder
from llama_index.core.node_parser import SentenceSplitter
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

load_dotenv()


splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=128,
)
# Dataset id from huggingface.co/dataset
dataset_id = "DataForGood/climateguard"
# Model id to load the tokenizer
model_id = "almanach/camembertav2-base"

# Load raw dataset
train_dataset = load_dataset(dataset_id, split="train")
records = []

for record in train_dataset:
    if record["misinformation_claims"]:
        for claim in record["misinformation_claims"]:
            for label in claim["labels"]:
                records.append({"text": claim["text"], "label": 1})
    else:
        chunks = splitter.split_text(record["plaintext"])
        records.append(
            {
                "text": random.choice(chunks),
                "label": int(record["misinformation"]),
            }
        )

claims_dataset = Dataset.from_pandas(pd.DataFrame.from_records(records))
split_dataset = claims_dataset.train_test_split(test_size=0.1)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)


# Tokenize helper function
def tokenize(batch):
    return tokenizer(
        batch["text"], padding=True, truncation=True, max_length=512, return_tensors="pt"
    )


def format_labels(example):
    return {"label": int(example["misinformation"])}


# Tokenize dataset
# if "misinformation" in split_dataset["train"].features.keys():
#     split_dataset =  split_dataset.rename_column("misinformation", "labels") # to match Trainer
tokenized_dataset = split_dataset.map(
    tokenize, batched=True, remove_columns=["text"]
)
# tokenized_dataset = tokenized_dataset.map(format_labels, batched=False)

# Prepare model labels - useful for inference
labels = list(set(tokenized_dataset["train"]["label"]))
num_labels = len(labels)
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# Download the model from huggingface.co/models
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=num_labels
)


# Metric helper method
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
        labels, predictions, labels=labels, pos_label=1, average="weighted"
    )
    return {"f1": float(score) if score == 1 else score}


# Define training args
training_args = TrainingArguments(
    output_dir=model_id.split("/")[-1] + "-" + dataset_id.split("/")[-1],
    run_name=model_id.split("/")[-1]
    + "-"
    + dataset_id.split("/")[-1]
    + datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    learning_rate=5e-6,
    warmup_steps=50,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    num_train_epochs=10,
    bf16=False,  # bfloat16 training
    fp16=False,
    optim="adamw_torch_fused",  # improved optimizer
    # logging & evaluation strategies
    logging_strategy="steps",
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    use_mps_device=True,
    metric_for_best_model="f1",
    # push to hub parameters
    push_to_hub=True,
    hub_strategy="every_save",
    hub_token=HfFolder.get_token(),
    report_to="wandb",
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)
trainer.train()
