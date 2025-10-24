import os
import uuid
from datetime import datetime

import torch
from datasets import ClassLabel, load_dataset
from sentence_transformers import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderClassificationEvaluator,
)
from sentence_transformers.cross_encoder.losses import CrossEntropyLoss
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")


WANDB = True

if wandb:
    print("reporting to wandb")
    wandb.login(key=os.getenv("WANDB_KEY"))


map_labels = {"Refutes": 0, "Supports": 1, "Not Enough Information": 2}

# Load an example training dataset that works with our loss function:
dataset = load_dataset("rabuahmad/climatecheck", split="train")
dataset = dataset.map(lambda row: {"label": row["annotation"]})
features = dataset.features.copy()
features["label"] = ClassLabel(
    names=["Refutes", "Supports", "Not Enough Information"], num_classes=3
)
dataset = dataset.map(features=features)
dataset = dataset.select_columns(["abstract", "claim", "label"])

split_data = dataset.train_test_split(test_size=0.1, stratify_by_column="label")
train_dataset, val_dataset = split_data["train"], split_data["test"]
split_data = train_dataset.train_test_split(test_size=0.1, stratify_by_column="label")
train_dataset, test_dataset = split_data["train"], split_data["test"]

model_name = "cross-encoder/nli-deberta-v3-base"
model = CrossEncoder(
    model_name, num_labels=3, activation_fn=torch.nn.functional.gelu
)  # num_labels=1 is for rerankers

def collate_fn(batch):
    inputs = [(sample["abstract"], sample["claim"]) for sample in batch]
    return (
        model.tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt", max_length=512
        ),
        torch.tensor([sample["label"] for sample in batch]),
    )


train_dataloader = DataLoader(
    train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

running_loss = 0.0
gradient_accumulation_steps = 4
validation_steps = 10
logging_steps = 5
for batch_idx, (batch, label) in enumerate(tqdm(train_dataloader)):
    batch = batch.to("mps")
    label = label.to("mps")
    out = model(**batch)
    loss = loss_fn(out.logits, label)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    running_loss += loss.item()
    del batch, label, out

    if ((batch_idx + 1) % logging_steps == 0):
        tqdm.write(f"Train Loss: {running_loss}")
        tqdm.write(f"Memory usage: {psutil.virtual_memory().percent}%")

    if ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_dataloader)):
        optimizer.step()
        optimizer.zero_grad()
        running_loss = 0.0
        torch.mps.empty_cache()
        tqdm.write(f"Memory usage: {psutil.virtual_memory().percent}%")

    if ((batch_idx + 1) % validation_steps == 0):
        validation_loss = 0
        predictions = []
        val_labels = []
        with torch.no_grad():
            for batch, label in tqdm(val_dataloader):
                val_labels.extend(label.detach().tolist())
                batch = batch.to("mps")
                label = label.to("mps")
                out = model(**batch).logits
                _pred = out.argmax(dim=1).detach().cpu().tolist()
                predictions.extend(_pred)
                loss = loss_fn(out, label)
                validation_loss += loss.item()
                del batch, label, out
                tqdm.write(f"Memory usage: {psutil.virtual_memory().percent}%")
            validation_loss /= len(val_dataloader)
            stats = {"eval_loss": validation_loss}
            stats.update(classification_report(val_labels, predictions, output_dict=True))
            tqdm.write(str(stats))
            torch.mps.empty_cache()



# # Load a model to train/finetune

# # Initialize the MultipleNegativesRankingLoss
# # This loss requires pairs of related texts or triplets
# loss = CrossEntropyLoss(model)

# dev_cls_evaluator = CrossEncoderClassificationEvaluator(
#     sentence_pairs=list(zip(val_dataset["claim"], val_dataset["abstract"])),
#     labels=val_dataset["label"],
#     name="climatecheck-dev",
# )
# dev_cls_evaluator(model)

# # 5. Define the training arguments
# short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
# run_name = f"classify-{short_model_name}-claims"
# output_dir = "output/" + run_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# train_batch_size = 4
# args = CrossEncoderTrainingArguments(
#     # Required parameter:
#     output_dir=output_dir,
#     # Optional training parameters:
#     num_train_epochs=5,
#     per_device_train_batch_size=train_batch_size,
#     per_device_eval_batch_size=train_batch_size,
#     warmup_ratio=0.1,
#     fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
#     bf16=False,  # Set to True if you have a GPU that supports BF16
#     # Optional tracking/debugging parameters:
#     eval_strategy="steps",
#     eval_steps=20,
#     save_strategy="steps",
#     save_steps=500,
#     save_total_limit=2,
#     logging_steps=10,
#     run_name=run_name,  # Will be used in W&B if `wandb` is installed
# )

# if WANDB:
#     run = wandb.init(
#         entity=os.getenv("WANDB_ENTITY", "gmguarino"),
#         project="claim_extraction",
#         config=args.to_dict(),
#         name=f"gmguarino/climateguard-{short_model_name}-claim-extraction-sft"
#         + str(uuid.uuid4()).split("-")[0],
#     )

# trainer = CrossEncoderTrainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     loss=loss,
#     evaluator=dev_cls_evaluator,
# )
# trainer.train()

# test_cls_evaluator = CrossEncoderClassificationEvaluator(
#     list(zip(test_dataset["claim"], test_dataset["abstract"])),
#     test_dataset["label"],
#     name="climatecheck-test",
# )
# print(test_cls_evaluator(model))
