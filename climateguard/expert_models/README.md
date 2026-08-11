# Expert Models

Per-topic fine-tuning datasets built from source PDFs (books/reports) that
debunk misinformation on a given subject. Each topic gets its own model
directory; a shared pipeline turns each source PDF into training
conversations.

## Layout

```
expert_models/
├── scripts/
│   ├── parse_pdf_to_jsonl.py       # PDF -> per-subsection JSONL
│   ├── generate_conversations.py   # per-subsection JSONL -> affirmation/debunk conversations
│   └── train_lora.py               # conversations -> LoRA fine-tuned model, pushed to the Hub
├── chat_templates/                 # optional Jinja chat templates for train_lora.py --chat-template
├── <model>/                        # e.g. biodiversity, insecurity
│   └── data/
│       ├── raw/                    # source PDF(s)
│       ├── parsed/                 # output of parse_pdf_to_jsonl.py
│       └── conversations/          # output of generate_conversations.py
├── train_output/                   # local adapter/merged-model output of train_lora.py (gitignored)
└── README.md
```

Each `<model>/data/` directory is gitignored (see `.gitignore`) — only the
scripts and this README are tracked. Source PDFs and generated data stay
local.

To add a new topic: create `expert_models/<model>/data/raw/` and drop the
source PDF(s) in it. Everything else is inferred from `<model>`.

## Pipeline

### 1. Parse the PDF into sections — `parse_pdf_to_jsonl.py`

Source PDFs generally have no usable outline/bookmarks, so chapters and
subsections are detected from font metadata instead of a table of contents
or blank lines (ordinary paragraph breaks look identical to real section
breaks, so `\n\n` is not a usable signal):

- **Chapters**: the largest heading-size cluster on a page starts a new
  chapter; a smaller cluster on the same page (if present) becomes a
  `kicker`/eyebrow label (e.g. "PREMIÈRE FAUSSE INFORMATION" above a chapter
  title). A page with only a kicker-tier heading (e.g. a part divider with no
  dedicated title font) still starts its own chapter, so its content isn't
  silently dropped.
- **Subsections**: within a chapter, a paragraph block set entirely in bold
  text at roughly body-text size marks a new subsection. Chapters whose PDF
  has no such convention yield a single untitled subsection spanning the
  whole chapter.

Output: one JSONL record per subsection in `data/parsed/<pdf_name>.jsonl`,
with `id`, `chapter_title`, `subsection_title`, `char_count`, `start_page`,
`end_page`, `text`.

```bash
# Defaults to the single PDF in data/raw and writes data/parsed/<pdf_name>.jsonl
uv run climateguard/expert_models/scripts/parse_pdf_to_jsonl.py biodiversity \
    --stop-after-keyword REMERCIEMENTS

uv run climateguard/expert_models/scripts/parse_pdf_to_jsonl.py insecurity
```

`--stop-after-keyword TEXT` drops all chapters after the first one whose
title/kicker matches TEXT (case-insensitive) — use it when the source has
trailing back matter (e.g. a duplicated table of contents, publisher
catalog) that isn't part of the book's content. Omit it if there's none.

See `--help` for `--pdf`/`--out` overrides.

### 2. Generate training conversations — `generate_conversations.py`

For each parsed subsection, a teacher LLM (Mistral by default, or Claude)
is asked to generate `--n-pairs` affirmation/debunk pairs grounded strictly
in that subsection's text:

- **Half TRUE** affirmations — statements the text supports. Debunk: a
  short, one-sentence confirmation.
- **Half FALSE** affirmations — myths/misconceptions the text contradicts.
  Debunk: a short explanation (2-4 sentences) citing concrete facts or
  figures from the text.

Output: one JSONL record per pair in `data/conversations/<pdf_name>.jsonl`,
as a two-turn conversation:

```json
{
  "messages": [
    {"role": "user", "content": "<affirmation>"},
    {"role": "assistant", "content": "<debunk>"}
  ],
  "metadata": {
    "is_true": true,
    "chapter_title": "...",
    "subsection_title": "...",
    "start_page": 12,
    "end_page": 14,
    "source_id": "02_01_..."
  }
}
```

```bash
# Requires MISTRAL_API_KEY (default provider) or ANTHROPIC_API_KEY in .env
uv run climateguard/expert_models/scripts/generate_conversations.py biodiversity

uv run climateguard/expert_models/scripts/generate_conversations.py insecurity \
    --provider claude --n-pairs 6
```

Useful flags:
- `--limit N` — process only the first N subsections (for a quick test).
- `--dry-run` — list what would be processed without calling the API.
- `--concurrency N` — max concurrent API calls (default 5).
- `--overwrite` — regenerate from scratch instead of resuming (by default,
  reruns skip subsections already present in the output file).

### 3. Fine-tune a model — `train_lora.py`

LoRA fine-tunes a base chat model on one or more topics' conversation files
with TRL's `SFTTrainer`, then pushes the result to the Hugging Face Hub as a
**private** model under the `DataForGood` org. Pairs are split into
train/eval by source excerpt (`source_id`), so pairs generated from the same
subsection never leak across the split.

Two independent choices:

- `--backend {accelerate,unsloth}` — training stack. `accelerate` (default)
  uses plain transformers + peft + trl and runs anywhere, including CPU/MPS.
  `unsloth` is faster/lighter but needs a CUDA GPU and the `unsloth` package.
- `--quant {4bit,none}` — `4bit` (default) performs real QLoRA via
  bitsandbytes NF4 quantization and requires a CUDA GPU + the `bitsandbytes`
  package. `none` is a plain LoRA over full/half-precision weights — the
  fallback for testing the whole pipeline locally on a laptop before
  spending GPU time.

`unsloth` and `bitsandbytes` live in the `cuda` optional dependency group,
since they're dead weight (and `unsloth` won't even import) on a non-CUDA
machine. On the GPU box, install them with:

```bash
uv sync --extra cuda
```

A topic-aware system prompt is built automatically (see
`build_system_prompt`/`TOPIC_DESCRIPTIONS` in the script — add an entry there
for each new topic) and prepended to every training example; override it
with `--system-prompt`. `--chat-template {default,chatml,mistral}` swaps in a
simple, known-good template from `chat_templates/*.jinja` instead of the
checkpoint's own — useful for keeping formatting (and assistant-only loss
masking via `{% generation %}` tags) consistent across different base
models. `default` uses the checkpoint's built-in template unchanged.

```bash
# Local smoke test on a laptop (CPU/MPS, tiny model, no push)
uv run climateguard/expert_models/scripts/train_lora.py biodiversity \
    --backend accelerate --quant none \
    --checkpoint Qwen/Qwen2.5-1.5B-Instruct --epochs 1 --limit 20

# Real QLoRA run on a GPU box, combining both topics, pushed to the Hub
uv run climateguard/expert_models/scripts/train_lora.py biodiversity insecurity \
    --backend unsloth --quant 4bit --push
```

The adapter is always saved locally to `train_output/adapter`. Pass `--push`
to also upload it (adapter-only) to `DataForGood/<checkpoint>-<topics>-qlora`
(or `-lora` when `--quant none`), or override with `--hub-repo`. Add
`--merge-and-push` to additionally merge the adapter into the base weights
and push a full deployable model — only supported with `--quant none`, since
merging a 4-bit-quantized base isn't (push the QLoRA adapter and merge it
against the full-precision base separately if you need a merged model from a
4-bit run).

See `--help` on any script for the full option list.
