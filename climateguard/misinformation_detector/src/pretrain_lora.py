"""
LoRA Continued Pre-Training (CPT) on climate reference material.

Pipeline:
  1. Load DataForGood/climateguard-training
  2. Extract all unique URLs from the debunk_references field
  3. Fetch each URL and extract clean text via trafilatura (async, cached)
  4. Optionally include the raw TV transcripts as additional CPT text
  5. Pack all text into fixed-length token chunks (standard CPT approach)
  6. Fine-tune with LoRA using a causal language modelling objective

The resulting adapter can be used as the starting point for SFT (finetune_lora.py).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import io
import re

import requests
import torch
import trafilatura
import yaml
from pypdf import PdfReader
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except (ImportError, Exception):
    UNSLOTH_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _apply_config(parser: argparse.ArgumentParser, config: dict) -> None:
    parser.set_defaults(**{k: v for k, v in config.items() if v is not None})


# ── URL extraction ────────────────────────────────────────────────────────────

def _parse_references(raw) -> list[str]:
    """Extract URLs from a debunk_references value (list, JSON string, or plain string)."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(u).strip() for u in raw if u and str(u).strip().startswith("http")]
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                items = json.loads(raw)
                return [str(u).strip() for u in items if u and str(u).strip().startswith("http")]
            except json.JSONDecodeError:
                pass
        if raw.startswith("http"):
            return [raw]
    return []


def collect_urls(ds_splits, country: str | None) -> list[str]:
    """Return a deduplicated list of reference URLs from all splits."""
    seen: set[str] = set()
    urls: list[str] = []
    for split in ds_splits:
        for row in split:
            if country and row.get("country") != country:
                continue
            for url in _parse_references(row.get("debunk_references")):
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
    log.info("Found %d unique reference URLs", len(urls))
    return urls


# ── Web fetching & text extraction ────────────────────────────────────────────

def _fetch_one(url: str) -> str | None:
    """Fetch a URL and extract the main text content using trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_recall=True,
        )
        return text.strip() if text and len(text.strip()) > 200 else None
    except Exception:
        return None


async def _fetch_async(url: str, semaphore: asyncio.Semaphore) -> tuple[str, str | None]:
    async with semaphore:
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, _fetch_one, url)
        await asyncio.sleep(0.3)  # polite crawl delay
        return url, text


async def fetch_all(urls: list[str], concurrency: int) -> dict[str, str]:
    """Fetch all URLs concurrently, returning {url: extracted_text}."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_fetch_async(url, semaphore) for url in urls]
    results: dict[str, str] = {}
    total = len(tasks)

    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        url, text = await coro
        if text:
            results[url] = text
        if i % 50 == 0 or i == total:
            log.info("Fetched %d / %d URLs (%d extracted)", i, total, len(results))

    return results


# ── PDF extraction ───────────────────────────────────────────────────────────

def _clean_pdf_text(text: str) -> str:
    """Remove common PDF artefacts: hyphenated line breaks, excessive whitespace."""
    text = re.sub(r"-\n(\w)", r"\1", text)        # re-join hyphenated words
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)   # collapse single newlines to spaces
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse 3+ newlines to paragraph break
    text = re.sub(r" {2,}", " ", text)              # collapse multiple spaces
    return text.strip()


# Matches French IPCC chapter headings such as:
#   "Chapitre 1" / "CHAPITRE 1" / "Résumé technique" / "Résumé à l'intention des décideurs"
_CHAPTER_RE = re.compile(
    r"(?:(?:Chapitre|CHAPITRE)\s+\d+|Résumé\s+(?:technique|à\s+l.intention))",
    re.IGNORECASE,
)


def _split_into_chapters(text: str) -> list[str]:
    """
    Split extracted PDF text into chapter-level documents.
    Falls back to returning the whole text as one document if no
    chapter headings are detected.
    """
    boundaries = [m.start() for m in _CHAPTER_RE.finditer(text)]
    if len(boundaries) < 2:
        log.warning(
            "No chapter headings detected — treating PDF as a single document. "
            "Consider adjusting _CHAPTER_RE if the PDF uses a different heading format."
        )
        return [text]

    chapters = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        chapter_text = text[start:end].strip()
        if len(chapter_text) > 200:
            chapters.append(chapter_text)

    log.info("Split PDF into %d chapters", len(chapters))
    return chapters


def fetch_pdf_chapters(url: str, cache_file: Path) -> list[str]:
    """
    Download a PDF, extract its text, split into chapters, and cache the result.
    Each chapter is returned as a separate document for CPT packing.
    The cache stores one chapter per line as a JSON record.
    """
    if cache_file.exists():
        log.info("Loading cached PDF chapters from %s", cache_file)
        chapters = []
        with cache_file.open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                chapters.append(obj["text"])
        log.info("Loaded %d cached chapters", len(chapters))
        return chapters

    log.info("Downloading PDF: %s", url)
    try:
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        log.error("Failed to download PDF: %s", e)
        return []

    log.info("Extracting text from PDF ...")
    try:
        reader = PdfReader(io.BytesIO(response.content))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(_clean_pdf_text(page_text))
        full_text = "\n\n".join(pages)
    except Exception as e:
        log.error("Failed to extract PDF text: %s", e)
        return []

    if len(full_text) < 500:
        log.warning("PDF extraction yielded very little text (%d chars) — skipping", len(full_text))
        return []

    chapters = _split_into_chapters(full_text)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as f:
        for i, ch in enumerate(chapters):
            f.write(json.dumps({"chapter": i + 1, "text": ch}, ensure_ascii=False) + "\n")
    log.info(
        "Cached %d chapters (%d total chars) to %s",
        len(chapters), sum(len(c) for c in chapters), cache_file,
    )
    return chapters


# ── Corpus building ───────────────────────────────────────────────────────────

def load_or_fetch_corpus(args, urls: list[str]) -> list[str]:
    """
    Return a list of text documents for CPT.
    If cache_file exists, load from it. Otherwise fetch, extract, and save.
    """
    cache_path = Path(args.cache_file)
    cached_urls: set[str] = set()
    texts: list[str] = []

    if cache_path.exists() and not args.overwrite_cache:
        log.info("Loading cached corpus from %s", cache_path)
        with cache_path.open() as f:
            for line in f:
                obj = json.loads(line)
                cached_urls.add(obj["url"])
                texts.append(obj["text"])
        log.info("Loaded %d cached documents", len(texts))

    missing = [u for u in urls if u not in cached_urls]
    if missing:
        log.info("Fetching %d uncached URLs ...", len(missing))
        fetched = asyncio.run(fetch_all(missing, args.fetch_concurrency))

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if cache_path.exists() and not args.overwrite_cache else "w"
        with cache_path.open(mode) as f:
            for url, text in fetched.items():
                f.write(json.dumps({"url": url, "text": text}, ensure_ascii=False) + "\n")
        texts.extend(fetched.values())
        log.info("Fetched %d new documents (total corpus: %d)", len(fetched), len(texts))
    else:
        log.info("All URLs already cached — skipping fetch")

    return texts


def add_explanations(ds_splits, country: str | None) -> list[str]:
    """Collect factchecker explanation texts from the dataset as CPT text."""
    texts = []
    for split in ds_splits:
        for row in split:
            if country and row.get("country") != country:
                continue
            raw = row.get("explanations")
            if not raw:
                continue
            # explanations is a list of strings or a single string
            items = raw if isinstance(raw, list) else [raw]
            for item in items:
                t = str(item).strip()
                if len(t) > 50:
                    texts.append(t)
    log.info("Added %d explanation texts to CPT corpus", len(texts))
    return texts


# ── Sequence packing ──────────────────────────────────────────────────────────

def pack_texts(texts: list[str], tokenizer, max_seq_len: int) -> Dataset:
    """
    Tokenize all texts, concatenate with EOS tokens, then split into
    fixed-length chunks of max_seq_len. This is the standard CPT packing
    approach — no padding, full efficiency.
    """
    log.info("Tokenising and packing %d documents ...", len(texts))
    eos = tokenizer.eos_token_id
    all_ids: list[int] = []

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        all_ids.append(eos)

    total_tokens = len(all_ids)
    n_chunks = total_tokens // max_seq_len
    log.info(
        "Total tokens: %d  →  %d chunks of %d",
        total_tokens, n_chunks, max_seq_len,
    )

    chunks = [
        all_ids[i * max_seq_len : (i + 1) * max_seq_len]
        for i in range(n_chunks)
    ]

    return Dataset.from_dict({"input_ids": chunks})


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(args):
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    if args.use_unsloth:
        log.info("Loading via Unsloth: %s", args.model)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_len,
            dtype=None,
            load_in_4bit=args.load_in_4bit,
            token=hf_token,
        )
    else:
        log.info("Loading via transformers: %s", args.model)
        bnb_config = None
        if args.load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif args.load_in_4bit:
            log.warning("4-bit quantization requires CUDA — loading in bfloat16 instead")

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if not bnb_config else None,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            token=hf_token,
        )
        model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def apply_lora(model, args):
    if args.use_unsloth:
        target_modules = [m.strip() for m in args.target_modules.split(",")]
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            target_modules=target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    else:
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


# ── Training ──────────────────────────────────────────────────────────────────

def train(args, dataset: Dataset, model, tokenizer):
    split = dataset.train_test_split(test_size=args.val_size, seed=42)

    gc_kwargs = {} if args.use_unsloth else {
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }

    training_args = TrainingArguments(
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
        report_to="wandb" if args.wandb else "none",
        run_name=args.run_name,
        **gc_kwargs,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=collator,
    )

    log.info(
        "CPT on %d chunks (train: %d, eval: %d)",
        len(dataset), len(split["train"]), len(split["test"]),
    )
    trainer.train()
    return trainer


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--env-file", default=".env")
    pre_args, _ = pre.parse_known_args()
    config = _load_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="LoRA continued pre-training on climate reference text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None,
                        help="YAML config file. CLI args override config values.")

    # Data
    parser.add_argument("--source-dataset", default="DataForGood/climateguard-training",
                        help="HuggingFace dataset to pull references and transcripts from")
    parser.add_argument("--country", default="france",
                        help="Filter to one country (default: france)")
    parser.add_argument("--include-explanations", action="store_true", default=True,
                        help="Include factchecker explanation texts as CPT text")
    parser.add_argument("--no-explanations", dest="include_explanations", action="store_false",
                        help="Exclude explanation texts from CPT corpus")
    parser.add_argument(
        "--ipcc-url",
        default="https://archive.ipcc.ch/pdf/assessment-report/ar5/wg1/WG1AR5_SummaryVolume_FINAL_FRENCH.pdf",
        help="URL of the IPCC French PDF to include in CPT corpus",
    )
    parser.add_argument(
        "--ipcc-cache", default="data/ipcc_french.jsonl",
        help="Path to cache the extracted IPCC chapters (default: data/ipcc_french.jsonl)",
    )
    parser.add_argument(
        "--include-ipcc", action="store_true", default=True,
        help="Include IPCC French report in CPT corpus",
    )
    parser.add_argument(
        "--no-ipcc", dest="include_ipcc", action="store_false",
        help="Exclude IPCC report from CPT corpus",
    )
    parser.add_argument("--cache-file", default="data/cpt_corpus.jsonl",
                        help="JSONL cache for fetched reference texts")
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="Re-fetch all URLs even if already cached")
    parser.add_argument("--fetch-concurrency", type=int, default=5,
                        help="Max concurrent URL fetches (default: 5)")

    # Model
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B-Instruct",
                        help="Base model on HuggingFace")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                        help="Disable 4-bit quantization (required on Mac)")
    parser.add_argument("--use-unsloth", action=argparse.BooleanOptionalAction, default=None,
                        help="Use Unsloth (auto when CUDA + unsloth available)")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules",
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Training
    parser.add_argument("--output-dir", default="output/olmo2-1b-cpt")
    parser.add_argument("--epochs", type=int, default=1,
                        help="CPT epochs — 1 is usually enough (default: 1)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate — lower than SFT to avoid forgetting (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--eval-steps", type=int, default=50)

    # Output
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--merge", action="store_true", default=True,
        help="Merge the LoRA adapter into the base weights after training and save "
             "the full model to --merged-output-dir. Required before SFT. (default: on)",
    )
    parser.add_argument(
        "--no-merge", dest="merge", action="store_false",
        help="Skip merging — save the adapter only.",
    )
    parser.add_argument(
        "--merged-output-dir", default=None,
        help="Where to save the merged model (default: <output-dir>-merged)",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", default="DataForGood/climateguard-olmo2-1b-cpt")
    parser.add_argument("--env-file", default=".env")

    _apply_config(parser, config)
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

    # Resolve Unsloth
    if args.use_unsloth is None:
        args.use_unsloth = UNSLOTH_AVAILABLE and torch.cuda.is_available()
    if args.use_unsloth and not UNSLOTH_AVAILABLE:
        log.error("--use-unsloth requested but unsloth is not installed.")
        sys.exit(1)

    device = get_device()
    log.info("Device: %s  |  Backend: %s", device, "Unsloth" if args.use_unsloth else "transformers")

    # ── Build CPT corpus ──────────────────────────────────────────────────────
    log.info("Loading source dataset: %s", args.source_dataset)
    raw_ds = load_dataset(args.source_dataset, token=hf_token)
    all_splits = list(raw_ds.values())

    urls = collect_urls(all_splits, args.country)
    corpus_texts = load_or_fetch_corpus(args, urls)

    if args.include_explanations:
        corpus_texts.extend(add_explanations(all_splits, args.country))

    if args.include_ipcc:
        ipcc_chapters = fetch_pdf_chapters(args.ipcc_url, Path(args.ipcc_cache))
        corpus_texts.extend(ipcc_chapters)

    if not corpus_texts:
        log.error("No text collected — check your data source and network access.")
        sys.exit(1)

    log.info("Total CPT documents: %d", len(corpus_texts))

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args)
    model = apply_lora(model, args)

    # ── Pack sequences ────────────────────────────────────────────────────────
    dataset = pack_texts(corpus_texts, tokenizer, args.max_seq_len)

    # ── Train ─────────────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer = train(args, dataset, model, tokenizer)

    # ── Save adapter ──────────────────────────────────────────────────────────
    log.info("Saving CPT adapter to %s", args.output_dir)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ── Merge ─────────────────────────────────────────────────────────────────
    if args.merge:
        merged_dir = Path(args.merged_output_dir or f"{args.output_dir}-merged")
        log.info("Merging LoRA weights into base model → %s", merged_dir)

        if args.use_unsloth:
            # Unsloth provides its own merge path that handles quantised weights correctly.
            merged_model = trainer.model.merge_and_unload()
        else:
            # For quantised (4-bit) models, reload in full precision before merging
            # because merge_and_unload() on a bnb model produces dequantised weights
            # that should be saved in bfloat16, not int4.
            if args.load_in_4bit and torch.cuda.is_available():
                log.info("Reloading base model in bfloat16 for clean merge ...")
                from peft import PeftModel
                base = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token,
                )
                merged_model = PeftModel.from_pretrained(base, args.output_dir)
                merged_model = merged_model.merge_and_unload()
            else:
                merged_model = trainer.model.merge_and_unload()

        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        log.info("Merged model saved to %s", merged_dir)
        log.info(
            "Use --model %s in finetune_lora.py to start SFT from this checkpoint.", merged_dir
        )

    # ── Push to Hub ───────────────────────────────────────────────────────────
    if args.push_to_hub:
        if not hf_token:
            log.error("--push-to-hub requires HUGGINGFACE_TOKEN in environment.")
            sys.exit(1)
        target = merged_model if args.merge else trainer.model
        log.info("Pushing to %s", args.hub_repo)
        target.push_to_hub(args.hub_repo, token=hf_token)
        tokenizer.push_to_hub(args.hub_repo, token=hf_token)
