"""Generate affirmation/debunk training conversations from parsed PDF sections.

Reads the per-subsection JSONL produced by parse_pdf_to_jsonl.py (chapter_title,
subsection_title, text, ...) and, for each subsection, asks a teacher LLM
(Claude or Mistral) to generate `--n-pairs` short affirmation/debunk pairs
grounded strictly in that subsection's text:

- Half are TRUE affirmations: statements the source text supports. Their
  debunk is a short sentence confirming the affirmation.
- Half are FALSE affirmations: statements the source text contradicts (myths,
  exaggerations, or misconceptions the text addresses). Their debunk is a
  short explanation of why it's false, citing facts/figures from the text.

Each pair is written as one two-turn conversation record:
    {"messages": [{"role": "user", "content": affirmation},
                   {"role": "assistant", "content": debunk}],
     "metadata": {"is_true": bool, "chapter_title": ..., "subsection_title": ...,
                  "start_page": ..., "end_page": ..., "source_id": ...}}

Usage:
    python generate_conversations.py <model> [--provider claude|mistral] [--n-pairs N]
        [--input PATH] [--output PATH] [--concurrency N] [--limit N]
        [--overwrite] [--dry-run] [--env-file .env]
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm.asyncio import tqdm as async_tqdm

EXPERT_MODELS_DIR = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = {
    "claude": "claude-sonnet-5",
    "mistral": "mistral-large-latest",
}

SYSTEM_PROMPT = """\
You produce training data for a fact-checking assistant, from an excerpt of a \
book that debunks misinformation on its topic using established facts, \
figures and scientific evidence.

Given the excerpt below, generate exactly {n_pairs} short affirmation/debunk \
pairs: {n_true} TRUE and {n_false} FALSE.

- TRUE affirmations are statements the excerpt supports. Their debunk is a \
short, single-sentence confirmation that the statement is accurate.
- FALSE affirmations are statements the excerpt contradicts or that \
misrepresent it - common myths, exaggerations or misconceptions on this \
topic. Their debunk is a short explanation (2-4 sentences) of why the \
affirmation is false, citing concrete facts or figures from the excerpt.

Hard rules:
- Base every affirmation and debunk strictly on the excerpt. Do not invent \
facts that are not in it.
- Write in the same language as the excerpt.
- An affirmation should read like something a person might plausibly say or \
believe, not like a quiz question or a direct quote from the text. Do not mention the extract.
- Output ONLY a JSON array, no prose before or after, no markdown code \
fences, matching this schema exactly:
[{{"affirmation": "...", "is_true": true, "debunk": "..."}}, ...]\
"""


def build_user_message(record: dict[str, Any]) -> str:
    heading = " - ".join(
        part for part in (record.get("chapter_title"), record.get("subsection_title")) if part
    )
    return f"EXCERPT ({heading}):\n{record['text']}"


def parse_pairs(raw: str, n_true: int, n_false: int) -> list[dict[str, Any]] | None:
    """Parse the teacher's JSON array response, tolerating markdown fences."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        pairs = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    cleaned = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        affirmation = str(pair.get("affirmation", "")).strip()
        debunk = str(pair.get("debunk", "")).strip()
        is_true = pair.get("is_true")
        if not affirmation or not debunk or not isinstance(is_true, bool):
            continue
        cleaned.append({"affirmation": affirmation, "is_true": is_true, "debunk": debunk})

    if not cleaned:
        return None
    return cleaned


# ── Providers ────────────────────────────────────────────────────────────────


async def call_claude(client, model: str, system: str, user: str) -> str:
    import anthropic

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.RateLimitError:
        await asyncio.sleep(30)
        return ""
    except anthropic.APIError:
        return ""
    return next((block.text for block in response.content if block.type == "text"), "")


async def call_mistral(client, model: str, system: str, user: str) -> str:
    response = await client.chat.complete_async(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def make_client(provider: str):
    if provider == "claude":
        import anthropic

        return anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if provider == "mistral":
        from mistralai.client import Mistral

        return Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    raise ValueError(f"Unknown provider: {provider}")


# ── Generation ───────────────────────────────────────────────────────────────


async def generate_for_record(
    client,
    provider: str,
    model: str,
    record: dict[str, Any],
    n_pairs: int,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    n_true = n_pairs // 2
    n_false = n_pairs - n_true
    system = SYSTEM_PROMPT.format(n_pairs=n_pairs, n_true=n_true, n_false=n_false)
    user = build_user_message(record)

    async with semaphore:
        call = call_claude if provider == "claude" else call_mistral
        raw = await call(client, model, system, user)

    pairs = parse_pairs(raw, n_true, n_false)
    if not pairs:
        return []

    results = []
    for pair in pairs:
        results.append(
            {
                "messages": [
                    {"role": "user", "content": pair["affirmation"]},
                    {"role": "assistant", "content": pair["debunk"]},
                ],
                "metadata": {
                    "is_true": pair["is_true"],
                    "chapter_title": record.get("chapter_title", ""),
                    "subsection_title": record.get("subsection_title", ""),
                    "start_page": record.get("start_page"),
                    "end_page": record.get("end_page"),
                    "source_id": record["id"],
                },
            }
        )
    return results


def load_records(input_path: Path) -> list[dict[str, Any]]:
    records = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def already_done_ids(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_id = obj.get("metadata", {}).get("source_id")
            if source_id:
                done.add(source_id)
    return done


async def main(args: argparse.Namespace) -> None:
    model_dir = EXPERT_MODELS_DIR / args.model
    parsed_dir = model_dir / "data" / "parsed"
    conversations_dir = model_dir / "data" / "conversations"

    if args.input:
        input_path = args.input
    else:
        jsonl_files = sorted(parsed_dir.glob("*.jsonl"))
        if not jsonl_files:
            sys.exit(f"No JSONL files found in {parsed_dir}")
        input_path = jsonl_files[0]

    if args.output:
        output_path = args.output
    else:
        conversations_dir.mkdir(parents=True, exist_ok=True)
        output_path = conversations_dir / input_path.name

    records = load_records(input_path)
    if args.limit:
        records = records[: args.limit]

    done = set() if args.overwrite else already_done_ids(output_path)
    if done:
        print(f"Resuming: {len(done)} source records already processed.")
    pending = [r for r in records if r["id"] not in done]
    print(f"Records to process: {len(pending)} / {len(records)}")

    if args.dry_run:
        for record in pending[: args.limit or len(pending)]:
            print(f"[dry-run] would generate {args.n_pairs} pairs for {record['id']}")
        return

    llm_model = args.model_name or DEFAULT_MODEL[args.provider]
    client = make_client(args.provider)
    semaphore = asyncio.Semaphore(args.concurrency)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    written = 0
    with output_path.open(mode, encoding="utf-8") as out_f:
        tasks = [
            generate_for_record(client, args.provider, llm_model, record, args.n_pairs, semaphore)
            for record in pending
        ]
        for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc=args.model):
            pairs = await coro
            for pair_record in pairs:
                out_f.write(json.dumps(pair_record, ensure_ascii=False) + "\n")
                written += 1
            out_f.flush()

    print(f"Wrote {written} conversation pairs to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Expert model directory name, e.g. 'biodiversity'")
    parser.add_argument(
        "--provider", choices=["claude", "mistral"], default="mistral", help="Teacher LLM provider"
    )
    parser.add_argument(
        "--model-name", default=None, help="Override the provider's default model id"
    )
    parser.add_argument("--n-pairs", type=int, default=4, help="Affirmation/debunk pairs per subsection")
    parser.add_argument("--input", type=Path, default=None, help="Input JSONL (default: single file in data/parsed)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL (default: data/conversations/<name>.jsonl)")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent API calls")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N source records")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output instead of resuming")
    parser.add_argument("--dry-run", action="store_true", help="List what would be processed without calling the API")
    parser.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")
    args = parser.parse_args()

    env_path = Path(args.env_file)
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    key_var = "ANTHROPIC_API_KEY" if args.provider == "claude" else "MISTRAL_API_KEY"
    if not args.dry_run and key_var not in os.environ:
        sys.exit(f"Error: {key_var} not set. Add it to your .env file or environment.")

    asyncio.run(main(args))
