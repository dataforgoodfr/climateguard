"""
Evaluate a fine-tuned LoRA model on the misinformation detection task.

Test data can be loaded from:
  - the source HF dataset  (DataForGood/climateguard-training, split=test)
  - the generated RCoT dataset (DataForGood/climate-misinformation-RCoT)
  - a local JSONL file produced by generate_synthetic_data.py --keep-metadata

Ground truth is derived from:
  - source dataset  → mesinfo_correct / mesinfo_incorrect / mesinfo_corrected_bool flags
  - generated JSONL → metadata.label field
  - RCoT HF dataset → metadata.label field
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except (ImportError, Exception):
    UNSLOTH_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Constants shared with the generation script ───────────────────────────────

MISINFO_TOKEN = "[[MISINFORMATION]]"
CLEAN_TOKEN = "[[CLEAN]]"

SOURCE_DATASET = "DataForGood/climateguard-training"

INFERENCE_SYSTEM = """\
You are a climate misinformation detector analysing raw TV transcripts.

For each transcript:
1. Identify any climate-related claims.
2. Assess whether those claims are misinformative.
3. End your response with [[MISINFORMATION]] if misinformation is present, or [[CLEAN]] if not.

Be concise. Most transcripts contain no misinformation — output [[CLEAN]] immediately when
nothing problematic is found. For detections, state the false claim and the correction in
dense factual sentences before the label.\
"""

# ── Ground-truth derivation (mirrors generate_synthetic_data.py) ──────────────

def derive_label(row: dict[str, Any]) -> str:
    if row["mesinfo_correct"]:
        return "MISINFORMATION"
    if not row["mesinfo_correct"] and row["mesinfo_incorrect"] and row["mesinfo_corrected_bool"]:
        return "MISINFORMATION"
    return "CLEAN"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_source_dataset(split: str, country: str | None, limit: int | None) -> list[dict]:
    """Load from DataForGood/climateguard-training — ground truth from annotation flags."""
    log.info("Loading source dataset %s  split=%s", SOURCE_DATASET, split)
    ds = load_dataset(SOURCE_DATASET, split=split)
    if country:
        ds = ds.filter(lambda r: r["country"] == country)
        log.info("Filtered to country='%s': %d examples", country, len(ds))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    records = []
    for row in ds:
        transcript = (row.get("data_item_plaintext") or "").strip()
        if not transcript:
            continue
        records.append({
            "transcript": transcript,
            "ground_truth": derive_label(row),
            "task_id": str(row.get("task_completion_aggregate_id", "")),
            "country": row.get("country", ""),
            "channel": row.get("data_item_channel_name", ""),
        })
    return records


def load_jsonl(path: str, limit: int | None) -> list[dict]:
    """Load from a local JSONL file produced with --keep-metadata."""
    log.info("Loading local file: %s", path)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("metadata", {})
            label = meta.get("label")
            if not label:
                log.warning("Record missing metadata.label — skipping. Re-generate with --keep-metadata.")
                continue
            msgs = obj.get("messages", [])
            transcript = next((m["content"] for m in msgs if m["role"] == "user"), "").strip()
            if not transcript:
                continue
            records.append({
                "transcript": transcript,
                "ground_truth": label,
                "task_id": meta.get("task_id", ""),
                "country": meta.get("country", ""),
                "channel": meta.get("channel", ""),
            })
            if limit and len(records) >= limit:
                break
    return records


def load_hf_generated(repo: str, split: str, limit: int | None) -> list[dict]:
    """Load from a generated RCoT HF dataset (DataForGood/climate-misinformation-RCoT)."""
    log.info("Loading HF generated dataset %s  split=%s", repo, split)
    ds = load_dataset(repo, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    records = []
    for row in ds:
        label = row.get("label")
        if not label:
            continue
        messages_raw = row.get("messages", "")
        if isinstance(messages_raw, str):
            msgs = json.loads(messages_raw)
        else:
            msgs = messages_raw
        transcript = next((m["content"] for m in msgs if m["role"] == "user"), "").strip()
        if not transcript:
            continue
        records.append({
            "transcript": transcript,
            "ground_truth": label,
            "task_id": str(row.get("task_id", "")),
            "country": row.get("country", ""),
            "channel": row.get("channel", ""),
        })
    return records


# ── Model loading ─────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(args) -> tuple:
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    device = get_device()

    use_unsloth = args.use_unsloth
    if use_unsloth is None:
        use_unsloth = UNSLOTH_AVAILABLE and torch.cuda.is_available()

    if use_unsloth:
        if not UNSLOTH_AVAILABLE:
            log.error("--use-unsloth requested but unsloth is not installed.")
            sys.exit(1)
        log.info("Loading via Unsloth: %s  adapter=%s", args.model, args.adapter)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_len,
            dtype=None,
            load_in_4bit=not args.no_4bit,
            token=hf_token,
        )
        if args.adapter:
            model = FastLanguageModel.get_peft_model(model)  # load adapter weights
            from peft import PeftModel as _PeftModel
            model = _PeftModel.from_pretrained(model, args.adapter)
        FastLanguageModel.for_inference(model)
    else:
        log.info("Loading via transformers: %s  adapter=%s", args.model, args.adapter)
        bnb_config = None
        if not args.no_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if not bnb_config else None,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            token=hf_token,
        )
        if device != "cuda":
            model = model.to(device)
        if args.adapter:
            log.info("Loading LoRA adapter: %s", args.adapter)
            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True, token=hf_token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    return model, tokenizer, device


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, device: str, transcript: str, args) -> str:
    messages = [
        {"role": "system", "content": INFERENCE_SYSTEM},
        {"role": "user", "content": transcript},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_len - args.max_new_tokens,
    )
    input_ids = inputs["input_ids"].to(device if device != "cuda" else next(model.parameters()).device)
    attention_mask = inputs["attention_mask"].to(input_ids.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_prediction(text: str) -> str | None:
    has_misinfo = MISINFO_TOKEN in text
    has_clean = CLEAN_TOKEN in text
    if has_misinfo and not has_clean:
        return "MISINFORMATION"
    if has_clean and not has_misinfo:
        return "CLEAN"
    if has_misinfo and has_clean:
        return "MISINFORMATION" if text.rfind(MISINFO_TOKEN) > text.rfind(CLEAN_TOKEN) else "CLEAN"
    return None


# ── Metrics ───────────────────────────────────────────────────────────────────

def print_metrics(ground_truths: list[str], predictions: list[str], unparseable: int) -> None:
    total = len(ground_truths)
    print(f"\n── Evaluation results {'─' * 40}")
    print(f"  Total evaluated : {total}  (unparseable: {unparseable})")

    labels = ["MISINFORMATION", "CLEAN"]
    print("\n" + classification_report(ground_truths, predictions, labels=labels, digits=3))

    cm = confusion_matrix(ground_truths, predictions, labels=labels)
    print("  Confusion matrix:")
    print(f"                     Pred MISINFO   Pred CLEAN")
    print(f"  GT  MISINFORMATION     {cm[0][0]:>5}         {cm[0][1]:>5}")
    print(f"  GT  CLEAN              {cm[1][0]:>5}         {cm[1][1]:>5}")
    print(f"{'─' * 60}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned misinformation detector.")

    # Data
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--source-dataset",
        action="store_true",
        help=f"Evaluate on {SOURCE_DATASET} (test split, human-annotated ground truth)",
    )
    data_group.add_argument(
        "--data-path",
        help="Local JSONL file (--keep-metadata required) or HF generated dataset ID",
    )
    parser.add_argument(
        "--hf-split", default="test",
        help="HF dataset split to use (default: test)",
    )
    parser.add_argument(
        "--country", default=None,
        help="Filter by country (e.g. france). Only applies to --source-dataset.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate on at most N examples",
    )

    # Model
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--adapter", default=None,
        help="Path to LoRA adapter directory or HF repo (optional — omit to evaluate the base model)",
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Disable 4-bit quantization (required on Mac)",
    )
    parser.add_argument(
        "--use-unsloth", action=argparse.BooleanOptionalAction, default=None,
        help="Use Unsloth for inference (auto-enabled when CUDA + unsloth available)",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=2048,
        help="Max input sequence length (default: 2048)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Max tokens to generate per example (default: 256)",
    )

    # Output
    parser.add_argument(
        "--output-file", default=None,
        help="Save per-example predictions to a JSONL file",
    )
    parser.add_argument("--env-file", default=".env")

    args = parser.parse_args()

    # ── Env ───────────────────────────────────────────────────────────────────
    env_path = Path(args.env_file)
    load_dotenv(env_path if env_path.exists() else None)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # ── Load test data ────────────────────────────────────────────────────────
    if args.source_dataset:
        records = load_source_dataset(args.hf_split, args.country, args.limit)
    else:
        path = Path(args.data_path)
        if path.exists() and path.is_file():
            records = load_jsonl(args.data_path, args.limit)
        else:
            records = load_hf_generated(args.data_path, args.hf_split, args.limit)

    if not records:
        log.error("No records loaded — check your data source and filters.")
        sys.exit(1)
    log.info("Loaded %d test records", len(records))

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer, device = load_model(args)
    log.info("Running inference on %s ...", device)

    # ── Inference loop ────────────────────────────────────────────────────────
    ground_truths, predictions, unparseable = [], [], 0
    output_rows = []

    for rec in tqdm(records, desc="Evaluating"):
        raw_output = run_inference(model, tokenizer, device, rec["transcript"], args)
        pred = parse_prediction(raw_output)

        if pred is None:
            log.warning("Could not parse label from: %r", raw_output[:120])
            unparseable += 1
            pred = "CLEAN"  # conservative fallback

        ground_truths.append(rec["ground_truth"])
        predictions.append(pred)

        if args.output_file:
            output_rows.append({
                "task_id": rec["task_id"],
                "country": rec["country"],
                "channel": rec["channel"],
                "ground_truth": rec["ground_truth"],
                "prediction": pred,
                "correct": pred == rec["ground_truth"],
                "raw_output": raw_output,
            })

    # ── Print metrics ─────────────────────────────────────────────────────────
    print_metrics(ground_truths, predictions, unparseable)

    # ── Save predictions ──────────────────────────────────────────────────────
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("Saved predictions to %s", args.output_file)
