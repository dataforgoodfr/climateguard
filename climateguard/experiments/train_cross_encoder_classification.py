from datetime import datetime

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

map_labels = {"Supports": 0, "Refutes": 1, "Not Enough Information": 2}

# Load an example training dataset that works with our loss function:
dataset = load_dataset("rabuahmad/climatecheck", split="train")
dataset = dataset.map(lambda row: {"label": row["annotation"]})
features = dataset.features.copy()
features["label"] = ClassLabel(
    names=["Supports", "Not Enough Information", "Refutes"], num_classes=3
)
dataset = dataset.map(features=features)
dataset = dataset.select_columns(["claim", "abstract", "label"])

# import sys; sys.exit(0)
split_data = dataset.train_test_split(test_size=0.1, stratify_by_column="label")
train_dataset, val_dataset = split_data["train"], split_data["test"]
split_data = train_dataset.train_test_split(test_size=0.1, stratify_by_column="label")
train_dataset, test_dataset = split_data["train"], split_data["test"]


model_name = "cross-encoder/nli-deberta-v3-base"
# Load a model to train/finetune
model = CrossEncoder(model_name, num_labels=3)  # num_labels=1 is for rerankers

# Initialize the MultipleNegativesRankingLoss
# This loss requires pairs of related texts or triplets
loss = CrossEntropyLoss(model)

dev_cls_evaluator = CrossEncoderClassificationEvaluator(
    sentence_pairs=list(zip(val_dataset["claim"], val_dataset["abstract"])),
    labels=val_dataset["label"],
    name="climatecheck-dev",
)
dev_cls_evaluator(model)

# 5. Define the training arguments
short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
run_name = f"classify-{short_model_name}-claims"
output_dir = "output/" + run_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
train_batch_size = 4
args = CrossEncoderTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=5,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=10,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=loss,
    evaluator=dev_cls_evaluator,
)
trainer.train()

# 7. Evaluate the final model on test dataset
test_cls_evaluator = CrossEncoderClassificationEvaluator(
    list(zip(test_dataset["claim"], test_dataset["abstract"])),
    test_dataset["label"],
    name="climatecheck-test",
)
print(test_cls_evaluator(model))
