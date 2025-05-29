import argparse
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from dotenv import load_dotenv
from huggingface_hub import HfFolder
from llama_index.core.node_parser import SentenceSplitter
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, labels=labels, pos_label=1, average="weighted")
    recall = recall_score(
        labels, predictions, labels=labels, pos_label=1, average="weighted"
    )
    precision = precision_score(
        labels, predictions, labels=labels, pos_label=1, average="weighted"
    )
    return {
        "f1": float(f1) if f1 == 1 else f1,
        "recall": float(recall) if recall == 1 else recall,
        "precision": float(precision) if precision == 1 else precision,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="DataForGood/climateguard",
        help="The HF id of the dataset used for training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="almanach/camembertav2-base",
        help="The HF id of the base model to finetune.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Data split to be trained on (default: 'train')",
    )
    parser.add_argument(
        "--train-val-split",
        type=float,
        default=0.1,
        help="Percentage of train set dedicated to validation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max segments length to train the model on (default: 512).",
    )
    parser.add_argument(
        "--text-feature",
        type=str,
        default="plaintext",
        help="Dataset feature containing the text input (default: 'plaintext').",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="misinformation",
        help="Dataset feature containing the labels (default: 'misinformation').",
    )
    parser.add_argument(
        "--chunk",
        type=argparse.BooleanOptionalAction,
        default=True,
        help="Split texts longer than max tokens into chunks (default: True)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size if chunk=True (default: 512).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=128,
        help="Overlap between chunks if chunk=True (default: 128).",
    )
    parser.add_argument(
        "--training-type",
        type=str,
        choices=["full", "out", "linear"],
        default="out",
        help=(
            "Type of finetuning. `full` corresponds to a full finetuning, "
            "`out` corresponds to training the classifier out layer only and "
            "`linear` trains both the classification linear layer as well as the linear pooler."
        ),
    )
    parser.add_argument(
        "--logging-type",
        type=str,
        choices=["None", "wandb"],
        default="wandb",
        help="Where to log to. Either `None` or `wandb` (for weights and biases)",
    )
    args = parser.parse_args()

    load_dotenv()
    if args.logging_type:
        wandb.login()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    splitter = SentenceSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Load raw dataset
    train_dataset = load_dataset(args.dataset, split=args.split)
    records = []

    # for record in train_dataset:
    #     # if record["misinformation_claims"]:
    #     #     for claim in record["misinformation_claims"]:
    #     #         for label in claim["labels"]:
    #     #             records.append({"text": claim["text"], "label": 1})
    #     chunks = splitter.split_text(record[args.text_feature])
    #     for chunk in chunks:
    #         records.append(
    #             {
    #                 "text": random.choice(chunk),
    #                 "label": int(record[args.label]),
    #             }
    #     )

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
    split_dataset = claims_dataset.train_test_split(test_size=args.train_val_split)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Tokenize helper function
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

    # Tokenize dataset
    tokenized_dataset = split_dataset.map(
        tokenize, batched=True, remove_columns=["text"]
    )

    # Prepare model labels - useful for inference
    labels = list(set(tokenized_dataset["train"]["label"]))
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Download the model from huggingface.co/models
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    # set all layers to not trainable
    if args.training_type != "full":
        for parameter in model.parameters():
            parameter.requires_grad = False

    if args.training_type == "out":
        # Set linear classifier to trainable
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    if args.training_type == "linear":
        for parameter in model.pooler.dense.parameters():
            parameter.requires_grad = True

    requires_grad_params = 0
    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.nelement()
        if parameter.requires_grad:
            requires_grad_params += parameter.nelement()
    print(
        f"Training {round(requires_grad_params/1e6)}M parameters out of {round(total_params/1e6)}M"
    )

    # Metric helper method

    # Define training args
    training_args = TrainingArguments(
        output_dir="runs/" + args.model.split("/")[-1] + "-" + args.dataset.split("/")[-1],
        run_name=args.model.split("/")[-1]
        + "-"
        + args.dataset.split("/")[-1]
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
        report_to=args.logging_type,
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

    for parameter in model.parameters():
        parameter.requires_grad = False

    print("Testing on validation set.")
    with torch.no_grad():
        # for record in tokenized_dataset:
        outputs = model(
            tokenized_dataset["test"]["input_ids"],
            tokenized_dataset["test"]["attention_mask"],
        )
    predictions = outputs.logits.argmax(1)
    labels = list(map(int, tokenized_dataset["test"]["misinformation"]))
    print(classification_report(labels, predictions))

    model.save_pretrained(
        "models/" + args.model.split("/")[-1] + "-" + args.dataset.split("/")[-1]
    )
    tokenizer.save_pretrained(
        "models/" + args.model.split("/")[-1] + "-" + args.dataset.split("/")[-1]
    )

    model.push_to_hub(args.model.split("/")[-1] + "-" + args.dataset.split("/")[-1])
    tokenizer.push_to_hub(args.model.split("/")[-1] + "-" + args.dataset.split("/")[-1])
