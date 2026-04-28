# climateguard · misinformation\_detector

Fine-tuning pipeline for a small open-weight LLM (Qwen 2.5 7B) that detects climate
misinformation in raw TV transcripts. A larger teacher model (Claude Sonnet 4.6) is used to
generate synthetic reasoning traces from existing factchecker annotations; those traces are
then used to fine-tune the small model for cheap inference.

See [`context.md`](context.md) for full design rationale.

---

## Setup

### Prerequisites

- [mise](https://mise.jdx.dev) — manages Python 3.13 and uv

### Install tools and dependencies

```bash
cd climateguard/misinformation_detector

# Install Python 3.13 and uv via mise
mise install

# Create virtualenv and install dependencies
uv sync
```

### Environment variables

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

`.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
HUGGINGFACE_TOKEN=hf_...      # only needed for --push-to-hub
```

---

## Synthetic data generation

`src/generate_synthetic_data.py` loads the
[`DataForGood/climateguard-training`](https://huggingface.co/datasets/DataForGood/climateguard-training)
dataset and calls Claude Sonnet 4.6 to produce a **reverse chain-of-thought** reasoning trace
for each example:

```
claims identified in transcript  →  why they are/aren't misinformation  →  [[MISINFORMATION]] or [[CLEAN]]
```

Output is a chat-format JSONL file ready for supervised fine-tuning:

```jsonc
{
  "messages": [
    {"role": "system", "content": "...inference system prompt..."},
    {"role": "user",   "content": "<raw TV transcript>"},
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

### Quick start

```bash
# Smoke-test on 20 examples, keep metadata columns in output
uv run python src/generate_synthetic_data.py \
  --limit 20 \
  --keep-metadata \
  --output data/test_traces.jsonl
```

### Full train split

```bash
uv run python src/generate_synthetic_data.py \
  --output data/synthetic_traces.jsonl
```

### Resume after interruption

The script tracks processed `task_id` values inside the output file and skips them on
re-runs — just rerun the same command:

```bash
uv run python src/generate_synthetic_data.py --output data/synthetic_traces.jsonl
```

### Process the test split

```bash
uv run python src/generate_synthetic_data.py \
  --split test \
  --output data/synthetic_traces_test.jsonl
```

### Push to HuggingFace Hub

Requires `HUGGINGFACE_TOKEN` in `.env` and write access to the target repo.

```bash
uv run python src/generate_synthetic_data.py \
  --output data/synthetic_traces.jsonl \
  --push-to-hub

# Custom repo
uv run python src/generate_synthetic_data.py \
  --output data/synthetic_traces.jsonl \
  --push-to-hub \
  --hub-repo your-org/your-dataset-name
```

The default target repo is `DataForGood/climate-misinformation-RCoT`.

### All options

```
--output          Output JSONL path            (default: data/synthetic_traces.jsonl)
--split           Dataset split: train | test  (default: train)
--limit           Process at most N examples   (default: all)
--concurrency     Max concurrent API calls     (default: 10)
--overwrite       Overwrite output file instead of resuming
--keep-metadata   Add task_id / label / cache stats columns to each record
--push-to-hub     Push to HuggingFace Hub after generation
--hub-repo        HuggingFace repo to push to  (default: DataForGood/climate-misinformation-RCoT)
--env-file        Path to .env file            (default: .env)
```

---

## LoRA fine-tuning

`src/finetune_lora.py` fine-tunes a causal LLM on the generated traces using QLoRA
(`bitsandbytes` 4-bit) and `SFTTrainer` from TRL. It accepts data from a local JSONL file
or directly from a HuggingFace dataset.

Tested on one NVIDIA L40S (48 GB). Default model: `Qwen/Qwen2.5-7B-Instruct`.

### From a local file

```bash
uv run python src/finetune_lora.py \
  --data-path data/synthetic_traces_france.jsonl \
  --output-dir output/qwen2.5-7b-lora
```

### From HuggingFace

```bash
uv run python src/finetune_lora.py \
  --data-path DataForGood/climate-misinformation-RCoT \
  --hf-split train \
  --output-dir output/qwen2.5-7b-lora
```

### With W&B logging and push to Hub

```bash
uv run python src/finetune_lora.py \
  --data-path data/synthetic_traces_france.jsonl \
  --output-dir output/qwen2.5-7b-lora \
  --wandb --run-name rcot-france-r16 \
  --push-to-hub \
  --hub-repo DataForGood/climateguard-qwen2.5-7b-lora
```

### Key options

```
Data
  --data-path       Local JSONL file or HF dataset ID            (required)
  --hf-split        HF split to use                              (default: train)
  --val-size        Fraction held out for validation             (default: 0.1)

Model
  --model           Base model on HuggingFace                    (default: Qwen/Qwen2.5-7B-Instruct)
  --load-in-4bit    QLoRA 4-bit quantization                     (default: on)
  --no-4bit         Disable 4-bit (run in bfloat16)
  --load-in-8bit    8-bit quantization instead

LoRA
  --lora-rank       LoRA rank r                                  (default: 16)
  --lora-alpha      LoRA alpha                                   (default: 32)
  --lora-dropout    LoRA dropout                                 (default: 0.05)
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
  --wandb           Log to W&B
  --push-to-hub     Push adapter to HuggingFace Hub after training
  --hub-repo        Target HF repo                               (default: DataForGood/climateguard-qwen2.5-7b-lora)
```

---

## Evaluation

`src/evaluate.py` runs the fine-tuned model on a test set and prints accuracy, precision,
recall, F1, and a confusion matrix. Ground truth is always sourced from human annotations —
never from the generated traces.

### Data sources

| Flag | Dataset | Ground truth |
|---|---|---|
| `--source-dataset` | `DataForGood/climateguard-training` test split | Derived from annotation flags |
| `--data-path <file.jsonl>` | Local JSONL (needs `--keep-metadata`) | `metadata.label` field |
| `--data-path DataForGood/climate-misinformation-RCoT` | HF generated dataset | `label` column |

### On the source dataset (recommended)

```bash
# Mac smoke test — base model, no adapter
uv run python src/evaluate.py \
  --source-dataset \
  --hf-split test \
  --country france \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --no-4bit \
  --limit 50

# With fine-tuned adapter
uv run python src/evaluate.py \
  --source-dataset \
  --hf-split test \
  --country france \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter output/qwen2.5-1.5b-lora \
  --no-4bit
```

### From a local generated file

```bash
uv run python src/evaluate.py \
  --data-path data/synthetic_traces_france.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter output/qwen2.5-1.5b-lora \
  --no-4bit
```

### From the HF generated dataset

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
  --source-dataset --hf-split test --country france \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter output/qwen2.5-1.5b-lora \
  --no-4bit \
  --output-file output/predictions.jsonl
```

### Key options

```
Data
  --source-dataset    Use DataForGood/climateguard-training (human-annotated)
  --data-path         Local JSONL or HF generated dataset ID
  --hf-split          Split to load                    (default: test)
  --country           Filter by country                (source-dataset only)
  --limit             Evaluate on at most N examples

Model
  --model             Base model                       (default: Qwen/Qwen2.5-1.5B-Instruct)
  --adapter           LoRA adapter path or HF repo     (omit to evaluate base model)
  --no-4bit           Disable quantization             (required on Mac)
  --use-unsloth /
  --no-unsloth        Force Unsloth on/off             (auto when CUDA + unsloth available)
  --max-new-tokens    Generation budget                (default: 256)

Output
  --output-file       Save per-example results to JSONL
```

---

## Cost estimate

The system prompt (~400 tokens) is cached after the first request, so subsequent calls pay
only ~0.1× of its token cost. At 10 concurrent workers, the full 5 009-example train split
takes roughly 15–20 minutes and costs approximately **$1–2** depending on average transcript
length.
