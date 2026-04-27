"""
Generate synthetic reasoning-trace training data for the climateguard misinformation detector.

Teacher model: claude-sonnet-4-6
Source dataset: DataForGood/climateguard-training

For each labeled example the teacher generates a concise chain-of-thought trace:
  claims identified  →  why they are/aren't misinformation  →  [[MISINFORMATION]] or [[CLEAN]]

The output is a chat-format JSONL file ready for supervised fine-tuning.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import anthropic
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as async_tqdm

# ── Labels ────────────────────────────────────────────────────────────────────

MISINFO_TOKEN = "[[MISINFORMATION]]"
CLEAN_TOKEN = "[[CLEAN]]"

# ── Prompts ───────────────────────────────────────────────────────────────────

# Used by the teacher at generation time.
TEACHER_SYSTEM = """\
You generate concise reasoning traces for training a climate misinformation detector.

You are given:
- A raw TV transcript
- Factchecker-identified claims extracted from it
- Factchecker explanations of the verdict
- The ground-truth label (MISINFORMATION or CLEAN)

Produce a reasoning trace written as if the detector itself is reading the transcript and
reasoning step by step to the verdict. The trace must be grounded only in the provided
annotations — do not invent facts.

Structure of the trace:
1. Briefly name the key claim(s) present (paraphrase from the transcript, guided by annotations).
2. For MISINFORMATION: state why the claim is false or misleading, citing the factchecker
   explanation in a single dense sentence. End with exactly: [[MISINFORMATION]]
3. For CLEAN: state in one sentence why the content is accurate or contains no climate misinfo.
   End with exactly: [[CLEAN]]

Hard rules:
- Maximum 4 sentences total for MISINFORMATION cases; 2 sentences for CLEAN.
- No hedging language ("seems", "appears to", "may be") — direct factual statements only.
- If misinformation was stated on air but then corrected by the presenter, still output
  [[MISINFORMATION]] — the false claim was broadcast.
- Output the trace only — no preamble, no meta-commentary.\
"""

# Embedded into every training example as the system prompt for the fine-tuned model.
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

# ── Label logic ───────────────────────────────────────────────────────────────


def derive_label(row: dict[str, Any]) -> bool:
    """Return True (MISINFORMATION) or False (CLEAN) from the dataset's counter-intuitive flags."""
    if row["mesinfo_correct"]:
        return True  # annotator correctly identified misinformation
    if not row["mesinfo_correct"] and row["mesinfo_incorrect"] and row["mesinfo_corrected_bool"]:
        return True  # misinfo was said on air, then corrected — still flag
    return False  # hard negative: no misinformation


# ── Teacher call ──────────────────────────────────────────────────────────────


async def generate_trace(
    client: anthropic.AsyncAnthropic,
    row: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Call the teacher to produce a reasoning trace for one dataset row."""
    is_misinfo = derive_label(row)
    label_str = "MISINFORMATION" if is_misinfo else "CLEAN"

    transcript = (row["data_item_plaintext"] or "").strip()
    if not transcript:
        return None

    claims_text = ""
    if row.get("claims"):
        claims = row["claims"]
        if isinstance(claims, list):
            claims_text = "\n".join(f"- {c}" for c in claims if c)
        elif isinstance(claims, str):
            claims_text = claims.strip()

    explanations_text = ""
    if row.get("explanations"):
        exps = row["explanations"]
        if isinstance(exps, list):
            explanations_text = "\n".join(f"- {e}" for e in exps if e)
        elif isinstance(exps, str):
            explanations_text = exps.strip()

    user_message = f"""\
TRANSCRIPT:
{transcript}

FACTCHECKER CLAIMS:
{claims_text or "(none provided)"}

FACTCHECKER EXPLANATIONS:
{explanations_text or "(none provided)"}

GROUND-TRUTH LABEL: {label_str}

Generate the reasoning trace.\
"""

    async with semaphore:
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=[
                    {
                        "type": "text",
                        "text": TEACHER_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.RateLimitError:
            await asyncio.sleep(30)
            return None
        except anthropic.APIError:
            return None

    trace = next((block.text for block in response.content if block.type == "text"), "").strip()

    if not trace:
        return None

    # Sanity check: ensure the expected label token appears.
    expected_token = MISINFO_TOKEN if is_misinfo else CLEAN_TOKEN
    if expected_token not in trace:
        trace = trace.rstrip(".") + f" {expected_token}"

    return {
        "messages": [
            {"role": "system", "content": INFERENCE_SYSTEM},
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": trace},
        ],
        "metadata": {
            "task_id": str(row.get("task_completion_aggregate_id", "")),
            "country": row.get("country", ""),
            "channel": row.get("data_item_channel_name", ""),
            "label": label_str,
            "cache_creation_tokens": response.usage.cache_creation_input_tokens,
            "cache_read_tokens": response.usage.cache_read_input_tokens,
        },
    }


# ── Validation ───────────────────────────────────────────────────────────────


def _extract_predicted_label(trace: str) -> str | None:
    """Return 'MISINFORMATION', 'CLEAN', or None if neither token is present."""
    has_misinfo = MISINFO_TOKEN in trace
    has_clean = CLEAN_TOKEN in trace
    if has_misinfo and not has_clean:
        return "MISINFORMATION"
    if has_clean and not has_misinfo:
        return "CLEAN"
    if has_misinfo and has_clean:
        # Whichever appears last is the model's final verdict.
        return (
            "MISINFORMATION" if trace.rfind(MISINFO_TOKEN) > trace.rfind(CLEAN_TOKEN) else "CLEAN"
        )
    return None


def validate_traces(output_path: Path) -> None:
    """Parse the generated JSONL and compare model labels to ground truth."""
    records = []
    with output_path.open() as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not records:
        print("Validation: no records found.", file=sys.stderr)
        return

    if not records[0].get("metadata"):
        print(
            "Validation skipped: no metadata in output file. Re-run with --keep-metadata.",
            file=sys.stderr,
        )
        return

    tp = tn = fp = fn = unparseable = 0

    for rec in records:
        gt_label = rec.get("metadata", {}).get("label")
        if not gt_label:
            unparseable += 1
            continue

        # Extract the assistant turn from the messages list.
        assistant_text = next(
            (m["content"] for m in rec.get("messages", []) if m["role"] == "assistant"),
            "",
        )
        pred_label = _extract_predicted_label(assistant_text)

        if pred_label is None:
            unparseable += 1
            continue

        if gt_label == "MISINFORMATION" and pred_label == "MISINFORMATION":
            tp += 1
        elif gt_label == "CLEAN" and pred_label == "CLEAN":
            tn += 1
        elif gt_label == "CLEAN" and pred_label == "MISINFORMATION":
            fp += 1
        elif gt_label == "MISINFORMATION" and pred_label == "CLEAN":
            fn += 1

    total = tp + tn + fp + fn
    if total == 0:
        print("Validation: no labelled records to evaluate.")
        return

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("\n── Label validation ─────────────────────────────────")
    print(f"  Total evaluated : {total}  (unparseable: {unparseable})")
    print(f"  Accuracy        : {accuracy:.3f}")
    print(f"  Precision       : {precision:.3f}  (of flagged, how many were real misinfo)")
    print(f"  Recall          : {recall:.3f}  (of real misinfo, how many were flagged)")
    print(f"  F1              : {f1:.3f}")
    print(f"\n  Confusion matrix:")
    print(f"                     Pred MISINFO   Pred CLEAN")
    print(f"  GT  MISINFORMATION     {tp:>5}         {fn:>5}")
    print(f"  GT  CLEAN              {fp:>5}         {tn:>5}")
    print("─────────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────


def push_to_hub(output_path: Path, split: str, repo_id: str, hf_token: str) -> None:
    """Load the generated JSONL and push it to the HuggingFace Hub."""
    records = []
    with output_path.open() as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not records:
        print("No records to push.", file=sys.stderr)
        return

    # Flatten messages list to individual string columns for HF compatibility,
    # keeping the raw messages JSON as a string column too.
    hf_rows = []
    for rec in records:
        msgs = rec.get("messages", [])
        row: dict[str, Any] = {"messages": json.dumps(msgs, ensure_ascii=False)}
        meta = rec.get("metadata", {})
        row.update(meta)
        hf_rows.append(row)

    ds = Dataset.from_list(hf_rows)
    ds_dict = DatasetDict({split: ds})
    ds_dict.push_to_hub(repo_id, token=hf_token)
    print(f"Pushed {len(hf_rows)} examples to {repo_id} (split: {split})")


async def main(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine already-processed IDs to allow resuming.
    done_ids: set[str] = set()
    if output_path.exists() and not args.overwrite:
        with output_path.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    tid = obj.get("metadata", {}).get("task_id", "")
                    if tid:
                        done_ids.add(tid)
                except json.JSONDecodeError:
                    pass
        print(f"Resuming: {len(done_ids)} examples already processed.")

    ds = load_dataset("DataForGood/climateguard-training")
    split = ds[args.split]

    if args.country:
        split = split.filter(lambda row: row["country"] == args.country)
        print(f"Filtered to country='{args.country}': {len(split)} examples")

    if args.limit:
        split = split.select(range(min(args.limit, len(split))))

    rows = [
        row for row in split if str(row.get("task_completion_aggregate_id", "")) not in done_ids
    ]
    print(f"Examples to process: {len(rows)}")

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    semaphore = asyncio.Semaphore(args.concurrency)

    # Track cache stats.
    total_cache_write = 0
    total_cache_read = 0
    written = 0

    # Validation requires metadata; enable it implicitly.
    save_metadata = args.keep_metadata or args.validate

    mode = "a" if (output_path.exists() and not args.overwrite) else "w"
    with output_path.open(mode) as out_f:
        tasks = [generate_trace(client, row, semaphore) for row in rows]

        for coro in async_tqdm.as_completed(tasks, total=len(tasks)):
            result = await coro
            if result is None:
                continue
            meta = result.get("metadata", {})
            total_cache_write += meta.get("cache_creation_tokens", 0)
            total_cache_read += meta.get("cache_read_tokens", 0)
            record = {"messages": result["messages"]}
            if save_metadata:
                record["metadata"] = meta
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1

    print(f"\nDone. Wrote {written} examples to {output_path}")
    print(f"Cache writes: {total_cache_write:,} tokens | Cache reads: {total_cache_read:,} tokens")

    if args.validate:
        validate_traces(output_path)

    if args.push_to_hub:
        hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if not hf_token:
            print(
                "Error: --push-to-hub requires HUGGINGFACE_TOKEN (or HF_TOKEN) in environment.",
                file=sys.stderr,
            )
            sys.exit(1)
        push_to_hub(output_path, args.split, args.hub_repo, hf_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training data traces.")
    parser.add_argument(
        "--output",
        default="data/synthetic_traces.jsonl",
        help="Output JSONL path (default: data/synthetic_traces.jsonl)",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--country",
        default="france",
        help="Filter to a single country value, e.g. france (default: all countries)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N examples (useful for testing)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent API calls (default: 10)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file instead of resuming",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Include metadata (task_id, label, cache stats) alongside messages in output",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="After generation, compare model labels to ground truth and print accuracy/F1",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the generated dataset to HuggingFace Hub after generation",
    )
    parser.add_argument(
        "--hub-repo",
        default="DataForGood/climate-misinformation-RCoT",
        help="HuggingFace repo to push to (default: DataForGood/climate-misinformation-RCoT)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    args = parser.parse_args()

    # Load .env before checking required variables.
    env_path = Path(args.env_file)
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # fall back to searching parent directories

    if "ANTHROPIC_API_KEY" not in os.environ:
        print(
            "Error: ANTHROPIC_API_KEY not set. Add it to your .env file or environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(main(args))
