# climateguard · misinformation\_detector

End-to-end pipeline for fine-tuning a small open-weight LLM to detect climate misinformation
in raw TV transcripts. A larger teacher model (Claude Sonnet 4.6) generates synthetic
reasoning traces from existing factchecker annotations; those traces are used to fine-tune
a small model (Qwen 2.5 1.5B or 7B) for cheap inference.

See [`context.md`](context.md) for full design rationale and hardware constraints.

---

## Project structure

```
misinformation_detector/
├── src/
│   ├── generate_synthetic_data.py   # Step 1 — teacher generates reasoning traces
│   ├── pretrain_lora.py             # Step 1b (optional) — LoRA-CPT on reference text
│   ├── finetune_lora.py             # Step 2 — LoRA fine-tuning on the traces
│   └── evaluate.py                  # Step 3 — evaluate on the test set
├── configs/
│   ├── generate_synthetic_data.yaml      # Data generation config (production)
│   ├── generate_synthetic_data_dev.yaml  # Data generation config (dev / smoke-test)
│   ├── pretrain_lora.yaml                # CPT config (production)
│   ├── pretrain_lora_dev.yaml            # CPT config (Mac / dev)
│   ├── finetune_lora.yaml                # SFT config (production)
│   ├── finetune_lora_dev.yaml            # SFT config (Mac / dev)
│   ├── evaluate.yaml                     # Evaluation config (production)
│   └── evaluate_dev.yaml                 # Evaluation config (Mac / dev)
├── data/                            # Generated JSONL files (git-ignored)
├── output/                          # LoRA adapters (git-ignored)
├── .env.example                     # API key template
├── mise.toml                        # Python 3.13 + uv via mise
├── pyproject.toml
└── context.md                       # Design doc
```

---

## Configuration files

All scripts accept a `--config` flag pointing to a YAML file.
Values in the config become defaults; any CLI argument overrides them.

```bash
# Use config as-is
uv run python src/finetune_lora.py --config configs/finetune_lora.yaml

# Override a single value from the config
uv run python src/finetune_lora.py --config configs/finetune_lora.yaml --epochs 5

# Override the data path without changing anything else
uv run python src/finetune_lora.py --config configs/finetune_lora.yaml \
  --data-path DataForGood/climate-misinformation-RCoT
```

YAML keys use the same names as the CLI `dest` (underscores, not hyphens):

```yaml
# configs/finetune_lora.yaml  — excerpt
model: Qwen/Qwen2.5-7B-Instruct
lora_rank: 16
lora_alpha: 16
epochs: 3
batch_size: 4
grad_accum: 4
lr: 2.0e-4
```

Set a value to `null` in YAML to let the script's built-in default take over.

---

## Setup

### Prerequisites

[mise](https://mise.jdx.dev) manages Python 3.13 and uv.

```bash
cd climateguard/misinformation_detector

# Install Python 3.13 and uv via mise
mise install

# Create virtualenv and install dependencies
uv sync
```

### Environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...       # required for data generation
HUGGINGFACE_TOKEN=hf_...           # required for gated models and --push-to-hub
WANDB_KEY=...                      # optional, only for --wandb
```

---

## Pipeline overview

```
DataForGood/climateguard-training
          │
          ├─ pretrain_lora.py (optional) ──────────────────────────────────┐
          │  1. fetch debunk_references URLs → trafilatura text extraction  │
          │  2. explanations field texts                                    │
          │  3. IPCC AR5 FR PDF → split by chapter                         │
          │  LoRA-CPT (causal LM objective)                                 │
          │  merge_and_unload() → full bfloat16 model                      │
          │                                                                 ▼
          │  generate_synthetic_data.py           output/olmo2-1b-cpt-merged/
          │  (Claude Sonnet 4.6 teacher)                                    │
          ▼                                                                 │
  data/synthetic_traces.jsonl                                              │
          │                                                                 │
          │  finetune_lora.py  ◄──────────── --model cpt-merged ───────────┘
          │  (QLoRA SFT, OLMo-2-1B / Qwen 2.5)
          ▼
  output/lora_adapter/
          │
          │  evaluate.py
          │  (source dataset test split)
          ▼
     accuracy / F1 / confusion matrix
```

---

## Step 1b (optional) — LoRA Continued Pre-Training

`src/pretrain_lora.py` adapts the base model to the climate/French domain *before* SFT.
Recommended when using an English-dominant base model like OLMo-2.

### Corpus sources

Three sources are combined, each independently cached:

| Source | Flag | Cache |
|---|---|---|
| Factcheck reference URLs (`debunk_references` field) | always on | `data/cpt_corpus.jsonl` |
| Factchecker explanation texts (`explanations` field) | `--include-explanations` (default: on) | inline |
| IPCC AR5 WG1 summary in French (PDF, split by chapter) | `--include-ipcc` (default: on) | `data/ipcc_french.jsonl` |

The IPCC PDF is split into chapters at heading boundaries (`Chapitre N`, `Résumé technique`, etc.) so each chapter is a separate document — chunks never straddle chapter boundaries during packing.

### Merge after training

After CPT, the LoRA adapter is **merged into the base weights** by default (`--merge`, on by default). Merging is required before SFT: `finetune_lora.py` loads a full model checkpoint, not a PEFT adapter directory. The merge reloads the base model in bfloat16 to avoid int4 artefacts from quantised weights.

```
output/olmo2-1b-cpt/         ← adapter weights (kept for inspection / rollback)
output/olmo2-1b-cpt-merged/  ← full bfloat16 model (pass to finetune_lora.py)
```

### Production run

```bash
uv run python src/pretrain_lora.py --config configs/pretrain_lora.yaml
```

### Mac smoke test

```bash
uv run python src/pretrain_lora.py --config configs/pretrain_lora_dev.yaml
```

### Use the merged model as the base for SFT

```bash
uv run python src/finetune_lora.py --config configs/finetune_lora.yaml \
  --model output/olmo2-1b-cpt-merged
```

### Override individual values

```bash
# Skip IPCC download, use explanations only
uv run python src/pretrain_lora.py --config configs/pretrain_lora.yaml --no-ipcc

# Re-fetch all URLs (ignores cache)
uv run python src/pretrain_lora.py --config configs/pretrain_lora.yaml --overwrite-cache

# Save merged model to a custom path
uv run python src/pretrain_lora.py --config configs/pretrain_lora.yaml \
  --merged-output-dir output/my-cpt-merged
```

### All options

```
Data
  --source-dataset        HF dataset to pull from              (default: DataForGood/climateguard-training)
  --country               Filter by country                    (default: france)
  --include-explanations  Include explanations field texts     (default: on)
  --no-explanations       Exclude explanation texts
  --include-ipcc          Include IPCC AR5 FR PDF              (default: on)
  --no-ipcc               Exclude IPCC report
  --ipcc-url              PDF URL                              (default: IPCC AR5 WG1 FR summary)
  --ipcc-cache            Cache path for extracted chapters    (default: data/ipcc_french.jsonl)
  --cache-file            JSONL cache for fetched URL texts    (default: data/cpt_corpus.jsonl)
  --overwrite-cache       Re-fetch all URLs even if cached
  --fetch-concurrency     Concurrent URL fetches               (default: 5)

Model / LoRA / Training
  (same flags as finetune_lora.py — see configs/pretrain_lora.yaml for CPT-specific defaults)
  --lr                    Learning rate — lower than SFT       (default: 1e-4)
  --epochs                CPT epochs — 1 is usually enough     (default: 1)

Output
  --merge / --no-merge    Merge adapter into base after training (default: on)
  --merged-output-dir     Where to save merged model           (default: <output-dir>-merged)
  --push-to-hub           Push merged model to HuggingFace Hub
  --hub-repo              Target repo                          (default: DataForGood/climateguard-olmo2-1b-cpt)
```

---

## Step 1 — Generate synthetic reasoning traces

`src/generate_synthetic_data.py` loads
[`DataForGood/climateguard-training`](https://huggingface.co/datasets/DataForGood/climateguard-training)
and calls Claude Sonnet 4.6 to produce a **reverse chain-of-thought** trace for each example:

```
claims identified in transcript  →  why they are/aren't misinformation  →  [[MISINFORMATION]] or [[CLEAN]]
```

Output is a chat-format JSONL ready for supervised fine-tuning:

```jsonc
{
  "messages": [
    {"role": "system",    "content": "...inference system prompt..."},
    {"role": "user",      "content": "<raw TV transcript>"},
    {"role": "assistant", "content": "The segment claims X. This is false because Y. [[MISINFORMATION]]"}
  ]
}
```

### Label semantics

The source dataset uses counterintuitive flag names — the script handles them correctly:

| `mesinfo_correct` | `mesinfo_incorrect` | `mesinfo_corrected_bool` | Training label |
|---|---|---|---|
| `True` | `False` | `*` | `[[MISINFORMATION]]` |
| `False` | `True` | `False` | `[[CLEAN]]` (hard negative) |
| `False` | `True` | `True` | `[[MISINFORMATION]]` (said on air, corrected — still flag) |

### Prompt caching

The teacher system prompt (~2 500 tokens of climate science reference + misinformation
patterns) is marked `cache_control: ephemeral`. Claude Sonnet 4.6 requires ≥ 2 048 tokens
to activate the cache; subsequent calls pay ~0.1× of the system-prompt token cost.

### Quick start

```bash
# Smoke-test: 20 French examples
uv run python src/generate_synthetic_data.py --config configs/generate_synthetic_data_dev.yaml

# Production: both splits, France, with validation
uv run python src/generate_synthetic_data.py --config configs/generate_synthetic_data.yaml
```

### Override individual values

```bash
# Different country
uv run python src/generate_synthetic_data.py --config configs/generate_synthetic_data.yaml \
  --country belgium

# Push to Hub after generation
uv run python src/generate_synthetic_data.py --config configs/generate_synthetic_data.yaml \
  --push-to-hub
```

### Resume after interruption

Re-run the same command — already-processed `task_id` values are skipped automatically.

### Validate teacher label accuracy

```bash
uv run python src/generate_synthetic_data.py \
  --country france \
  --keep-metadata \
  --validate \
  --output data/synthetic_traces_france.jsonl
```

Sample output:

```
── Label validation ─────────────────────────────────
  Total evaluated : 3201  (unparseable: 0)
  Accuracy        : 0.961
  Precision       : 0.887
  Recall          : 0.923
  F1              : 0.905
─────────────────────────────────────────────────────
```

### Push a single split to HuggingFace Hub

```bash
uv run python src/generate_synthetic_data.py \
  --country france \
  --output data/synthetic_traces_france.jsonl \
  --push-to-hub \
  --hub-repo DataForGood/climate-misinformation-RCoT
```

### Generate both splits and push as a full DatasetDict

Use `--all-splits` to process train and test in one command. Output paths are derived
automatically from `--output` by injecting the split name into the stem:

```
data/synthetic_traces_france.jsonl
  → data/synthetic_traces_france_train.jsonl
  → data/synthetic_traces_france_test.jsonl
```

```bash
uv run python src/generate_synthetic_data.py \
  --country france \
  --all-splits \
  --output data/synthetic_traces_france.jsonl \
  --push-to-hub \
  --hub-repo DataForGood/climate-misinformation-RCoT
```

Both splits are pushed as a single `DatasetDict` so the resulting HF dataset has the
standard `train` / `test` structure. Resume works per-split independently.

### All options

```
--output          Output JSONL path (base when --all-splits)   (default: data/synthetic_traces.jsonl)
--split           Dataset split: train | test                  (default: train, ignored with --all-splits)
--all-splits      Process both train and test splits
--country         Filter to one country                        (default: france)
--limit           Process at most N examples per split         (default: all)
--concurrency     Max concurrent API calls                     (default: 10)
--overwrite       Overwrite output files instead of resuming
--keep-metadata   Add task_id / label / cache stats to each record
--validate        Compare model labels to ground truth after each split
--push-to-hub     Push to HuggingFace Hub after generation
--hub-repo        HF repo                                      (default: DataForGood/climate-misinformation-RCoT)
--env-file        Path to .env file                            (default: .env)
```

---

## Step 2 — LoRA fine-tuning

`src/finetune_lora.py` fine-tunes a causal LLM on the generated traces using QLoRA
(`bitsandbytes` 4-bit) and `SFTTrainer` from TRL. Data can come from a local JSONL file
or directly from a HuggingFace dataset.

**Recommended model:** `Qwen/Qwen2.5-1.5B-Instruct` for a ~1 B experiment,
`Qwen/Qwen2.5-7B-Instruct` for production. Both share the same tokenizer and LoRA target
modules. Tested on one NVIDIA L40S (48 GB).

**Unsloth** is auto-enabled when CUDA is available and `unsloth` is installed
(`uv sync --extra train`). Pass `--no-unsloth` to force the standard transformers path.

### Production run (7B, L40S)

```bash
uv run python src/finetune_lora.py --config configs/finetune_lora.yaml
```

### Mac smoke test (no CUDA)

```bash
uv run python src/finetune_lora.py --config configs/finetune_lora_dev.yaml
```

### Override individual values

```bash
# Different data source, everything else from config
uv run python src/finetune_lora.py --config configs/finetune_lora.yaml \
  --data-path DataForGood/climate-misinformation-RCoT

# Longer run
uv run python src/finetune_lora.py --config configs/finetune_lora.yaml --epochs 5
```

### With W&B logging and push to Hub

```bash
uv run python src/finetune_lora.py \
  --data-path data/synthetic_traces_france.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir output/qwen2.5-1.5b-lora \
  --wandb --run-name rcot-france-r16 \
  --push-to-hub \
  --hub-repo DataForGood/climateguard-qwen2.5-1.5b-lora
```

### All options

```
Data
  --data-path       Local JSONL file or HF dataset ID            (required)
  --hf-split        HF split to use                              (default: train)
  --val-size        Fraction held out for validation             (default: 0.1)

Model
  --model           Base model on HuggingFace                    (default: Qwen/Qwen2.5-7B-Instruct)
  --load-in-4bit    QLoRA 4-bit quantization                     (default: on)
  --no-4bit         Disable 4-bit (run in bfloat16, required on Mac)
  --load-in-8bit    8-bit quantization instead of 4-bit
  --use-unsloth /
  --no-unsloth      Force Unsloth on/off (auto when CUDA + unsloth installed)

LoRA
  --lora-rank       LoRA rank r                                  (default: 16)
  --lora-alpha      LoRA alpha                                   (default: 32)
  --lora-dropout    LoRA dropout                                 (default: 0.05, 0 for Unsloth)
  --target-modules  Comma-separated module names                 (default: q/k/v/o + gate/up/down proj)

Training
  --output-dir      Adapter save path                            (default: output/lora_adapter)
  --epochs          Training epochs                              (default: 3)
  --batch-size      Per-device batch size                        (default: 2)
  --grad-accum      Gradient accumulation steps                  (default: 8)
  --lr              Learning rate                                (default: 2e-4)
  --max-seq-len     Max sequence length                          (default: 2048)
  --eval-steps      Evaluate every N steps                       (default: 50)

Output
  --wandb           Log to Weights & Biases
  --run-name        W&B run name
  --push-to-hub     Push adapter to HuggingFace Hub after training
  --hub-repo        Target HF repo                               (default: DataForGood/climateguard-qwen2.5-7b-lora)
  --env-file        Path to .env file                            (default: .env)
```

---

## Step 3 — Evaluation

`src/evaluate.py` runs the fine-tuned model on a test set and prints accuracy, precision,
recall, F1, and a confusion matrix. Ground truth always comes from human annotations —
never from the generated traces.

### Data sources

| Flag | Dataset | Ground truth |
|---|---|---|
| `--source-dataset` | `DataForGood/climateguard-training` test split | Derived from annotation flags (same logic as generation) |
| `--data-path <file.jsonl>` | Local JSONL (requires `--keep-metadata` at generation time) | `metadata.label` field |
| `--data-path <hf-repo>` | HF generated dataset (e.g. `DataForGood/climate-misinformation-RCoT`) | `label` column |

### Production evaluation

```bash
uv run python src/evaluate.py --config configs/evaluate.yaml
```

### Mac smoke test

```bash
uv run python src/evaluate.py --config configs/evaluate_dev.yaml
```

### Override individual values

```bash
# Evaluate a different adapter
uv run python src/evaluate.py --config configs/evaluate.yaml \
  --adapter output/my-other-adapter

# Evaluate base model (no adapter)
uv run python src/evaluate.py --config configs/evaluate.yaml --adapter null
```

### Evaluate from a local generated file

```bash
uv run python src/evaluate.py \
  --data-path data/synthetic_traces_france.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter output/qwen2.5-1.5b-lora \
  --no-4bit
```

### Evaluate from the HF generated dataset

```bash
uv run python src/evaluate.py \
  --data-path DataForGood/climate-misinformation-RCoT \
  --hf-split test \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter output/qwen2.5-1.5b-lora \
  --no-4bit
```

### Save per-example predictions

```bash
uv run python src/evaluate.py \
  --source-dataset \
  --hf-split test \
  --country france \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter output/qwen2.5-1.5b-lora \
  --no-4bit \
  --output-file output/predictions.jsonl
```

Sample output:

```
── Evaluation results ────────────────────────────────────────────────
                precision    recall  f1-score   support

MISINFORMATION      0.881     0.904     0.892       312
         CLEAN      0.961     0.951     0.956       893

      accuracy                          0.939      1205

  Confusion matrix:
                     Pred MISINFO   Pred CLEAN
  GT  MISINFORMATION       282            30
  GT  CLEAN                 44           849
──────────────────────────────────────────────────────────────────────
```

### All options

```
Data
  --source-dataset    Use DataForGood/climateguard-training (mutually exclusive with --data-path)
  --data-path         Local JSONL or HF dataset ID          (mutually exclusive with --source-dataset)
  --hf-split          Split to load                         (default: test)
  --country           Filter by country                     (source-dataset only, e.g. france)
  --limit             Evaluate on at most N examples

Model
  --model             Base model                            (default: Qwen/Qwen2.5-1.5B-Instruct)
  --adapter           LoRA adapter path or HF repo          (omit to evaluate the base model)
  --no-4bit           Disable quantization                  (required on Mac)
  --use-unsloth /
  --no-unsloth        Force Unsloth on/off                  (auto when CUDA + unsloth available)
  --max-seq-len       Max input sequence length             (default: 2048)
  --max-new-tokens    Generation token budget               (default: 256)

Output
  --output-file       Save per-example results to JSONL
  --env-file          Path to .env file                     (default: .env)
```

---

## Cost estimate

The teacher system prompt (~2 500 tokens) is cached after the first request. At 10 concurrent
workers, the full 3 200-example French train split takes roughly 10–15 minutes and costs
approximately **$0.50–1.00** depending on average transcript length.
