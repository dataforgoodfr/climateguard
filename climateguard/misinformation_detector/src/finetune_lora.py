"""
LoRA fine-tuning of a causal LLM on the generated reasoning-trace dataset.

Data can be loaded from:
  - a local JSONL file  (--data-path ./data/synthetic_traces.jsonl)
  - a HuggingFace dataset (--data-path DataForGood/climate-misinformation-RCoT)

The expected record format (same as generate_synthetic_data.py output):
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

When loaded from HF the "messages" column may be a JSON string — both forms are handled.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except (ImportError):
    UNSLOTH_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ── Device ────────────────────────────────────────────────────────────────────


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Data loading ──────────────────────────────────────────────────────────────


def _normalise_messages(raw) -> list[dict]:
    """Accept messages as a Python list or a JSON string (HF Hub storage format)."""
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def load_data(data_path: str, hf_split: str, val_size: float) -> DatasetDict:
    """
    Load from a local JSONL file or a HuggingFace dataset ID.
    Returns a DatasetDict with 'train' and 'validation' splits.
    """
    path = Path(data_path)

    if path.exists() and path.is_file():
        log.info("Loading data from local file: %s", data_path)
        records = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append({"messages": _normalise_messages(obj["messages"])})
        ds = Dataset.from_list(records)
    else:
        log.info("Loading HuggingFace dataset: %s  split=%s", data_path, hf_split)
        raw = load_dataset(data_path, split=hf_split)
        ds = raw.map(
            lambda ex: {"messages": _normalise_messages(ex["messages"])},
            desc="Normalising messages",
        )
        ds = ds.select_columns(["messages"])

    split = ds.train_test_split(test_size=val_size, seed=42)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


# ── Formatting ────────────────────────────────────────────────────────────────


def make_formatting_func(tokenizer):
    def formatting_func(examples):
        chats = [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            for msgs in examples["messages"]
        ]
        return {"text": chats}

    return formatting_func


# ── Model ─────────────────────────────────────────────────────────────────────


def load_model_and_tokenizer(args):
    log.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        token=os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = None
    if args.load_in_4bit:
        log.info("Using 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        log.info("Using 8-bit quantization")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    log.info("Loading model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"),
    )
    model.config.use_cache = False

    return model, tokenizer


# ── LoRA ──────────────────────────────────────────────────────────────────────


def apply_lora(model, args):
    target_modules = [m.strip() for m in args.target_modules.split(",")]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Unsloth model + LoRA ─────────────────────────────────────────────────────


def load_model_and_tokenizer_unsloth(args):
    log.info("Loading model + tokenizer via Unsloth: %s", args.model)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,  # auto-detect bfloat16 / float16
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def apply_lora_unsloth(model, args):
    target_modules = [m.strip() for m in args.target_modules.split(",")]
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,          # Unsloth only supports dropout=0
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )
    model.print_trainable_parameters()
    return model


# ── Training ──────────────────────────────────────────────────────────────────


def train(args, dataset: DatasetDict, model, tokenizer, use_unsloth: bool = False):
    formatting_func = make_formatting_func(tokenizer)

    # Pre-format the dataset so SFTTrainer sees a plain "text" column.
    dataset = dataset.map(formatting_func, batched=True, desc="Applying chat template")

    # Unsloth manages gradient checkpointing via use_gradient_checkpointing="unsloth"
    # set in get_peft_model — passing it again in SFTConfig causes a conflict.
    gc_kwargs = {} if use_unsloth else {
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=False,
        **gc_kwargs,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        report_to="wandb" if args.wandb else "none",
        run_name=args.run_name,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    log.info(
        "Training on %d examples, validating on %d",
        len(dataset["train"]),
        len(dataset["validation"]),
    )
    trainer.train()
    return trainer


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA fine-tune a causal LLM on reasoning traces.")

    # Data
    parser.add_argument(
        "--data-path",
        required=True,
        help="Local JSONL file or HuggingFace dataset ID",
    )
    parser.add_argument(
        "--hf-split",
        default="train",
        help="Split to use when loading from HuggingFace (default: train)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of data to hold out as validation (default: 0.1)",
    )

    # Model
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model id on HuggingFace (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="QLoRA: load model in 4-bit (default: True)",
    )
    parser.add_argument(
        "--no-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit quantization",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit instead of 4-bit",
    )
    parser.add_argument(
        "--use-unsloth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use Unsloth for faster training. "
            "Auto-enabled when CUDA is available and unsloth is installed. "
            "Pass --no-unsloth to force the standard transformers path."
        ),
    )

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank r (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)"
    )
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules",
    )

    # Training
    parser.add_argument(
        "--output-dir", default="output/lora_adapter", help="Where to save the adapter"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Per-device batch size (default: 2)"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=8, help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.3, help="Max gradient norm (default: 0.3)"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=2048, help="Max sequence length (default: 2048)"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=50, help="Evaluate every N steps (default: 50)"
    )

    # Logging / hub
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--run-name", default=None, help="W&B run name")
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push adapter to HuggingFace Hub after training"
    )
    parser.add_argument(
        "--hub-repo",
        default="DataForGood/climateguard-qwen2.5-7b-lora",
        help="HF repo to push adapter to (default: DataForGood/climateguard-qwen2.5-7b-lora)",
    )
    parser.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")

    args = parser.parse_args()

    # ── Env ───────────────────────────────────────────────────────────────────
    env_path = Path(args.env_file)
    load_dotenv(env_path if env_path.exists() else None)

    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    if args.wandb:
        import wandb

        wandb.login(key=os.environ.get("WANDB_KEY"))

    if args.load_in_8bit:
        args.load_in_4bit = False

    device = get_device()
    log.info("Device: %s", device)
    log.info(
        "bfloat16 supported: %s",
        torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    )

    # Resolve --use-unsloth: auto-enable when CUDA available and package installed.
    if args.use_unsloth is None:
        args.use_unsloth = UNSLOTH_AVAILABLE and torch.cuda.is_available()
    if args.use_unsloth and not UNSLOTH_AVAILABLE:
        log.error("--use-unsloth requested but unsloth is not installed. "
                  "Install it with: pip install unsloth")
        sys.exit(1)
    if args.use_unsloth and not torch.cuda.is_available():
        log.error("--use-unsloth requires a CUDA GPU.")
        sys.exit(1)
    log.info("Backend: %s", "Unsloth" if args.use_unsloth else "transformers + peft")

    # ── Load data ─────────────────────────────────────────────────────────────
    dataset = load_data(args.data_path, args.hf_split, args.val_size)
    log.info("Train: %d  |  Validation: %d", len(dataset["train"]), len(dataset["validation"]))

    # ── Load model ────────────────────────────────────────────────────────────
    if args.use_unsloth:
        model, tokenizer = load_model_and_tokenizer_unsloth(args)
        model = apply_lora_unsloth(model, args)
    else:
        model, tokenizer = load_model_and_tokenizer(args)
        model = apply_lora(model, args)

    # ── Train ─────────────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer = train(args, dataset, model, tokenizer, use_unsloth=args.use_unsloth)

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info("Saving adapter to %s", args.output_dir)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        if not hf_token:
            log.error("--push-to-hub requires HUGGINGFACE_TOKEN in environment.")
            sys.exit(1)
        log.info("Pushing adapter to %s", args.hub_repo)
        trainer.model.push_to_hub(args.hub_repo, token=hf_token)
        tokenizer.push_to_hub(args.hub_repo, token=hf_token)
        log.info("Done.")
