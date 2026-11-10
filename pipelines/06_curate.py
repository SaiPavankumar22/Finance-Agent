"""Run CuratorKIT and project-specific gates over the WS4 gold set.

CuratorKIT performs the real schema, deduplication, and cleaning stages. The
post-checks below enforce this project's trajectory contract because generic
dataset curation cannot know the finance tool registry.

Two settings matter:

* ``min_tokens=0``. CuratorKIT's schema gate counts only the primary text of a
  sample, which for a detected conversation is the user question plus the first
  assistant call (always a single JSON token). Its default minimum of 10 tokens
  therefore rejected short but perfectly valid questions and emptied nine of
  the thirteen intents. Trajectory length is checked below instead.
* Metadata is re-attached after export. CuratorKIT's messages exporter writes
  ``{"messages": [...]}`` only, so ``id``/``intent``/``tickers``/``period`` are
  joined back from the gold file by exact question text.

Usage:
    python pipelines/06_curate.py
    python pipelines/06_curate.py --max-samples 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from curatorkit import Curator, CuratorConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "data" / "gold" / "trajectories.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "curated"
CURATORKIT_OUTPUT = OUTPUT_DIR / "sft_messages.jsonl"
VALIDATED = OUTPUT_DIR / "validated_messages.jsonl"
CURATED = OUTPUT_DIR / "curated.jsonl"
GATE_LOG = OUTPUT_DIR / "gate_counts.json"
MANIFEST = OUTPUT_DIR / "manifest.json"
CHECKSUMS = OUTPUT_DIR / "checksums.txt"
ALLOWED_TOOLS = {
    "get_stock_quote",
    "get_financial_ratios",
    "get_historical_prices",
    "get_company_news",
    "finish",
}
PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y"}
ADVICE_WORDS = re.compile(r"\b(buy|sell|guaranteed|will double)\b", re.I)
# Exact de-duplication, not MinHash. The gold planner already guarantees unique
# question text, and near-duplicate suppression at 0.85 collapses questions that
# differ only by their ticker (for example "Summarise WXYZ for me."), which is
# exactly the per-ticker coverage the train/val split and the invalid-ticker
# intent depend on. MinHash removed 327 of 800 rows for that reason.
DEDUP = "exact"
MINHASH_THRESHOLD = 0.85
USABLE_STATUSES = {"ok", "stale"}
MAX_ATTEMPTS_PER_CALL = 2
METADATA_KEYS = (
    "id",
    "intent",
    "mutation",
    "language",
    "tickers",
    "period",
    "question",
    "expected_behavior",
    "report",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_json(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def call_sequence(
    messages: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str | None]]:
    """Pair each assistant call with the status of the result it returned."""
    pairs: list[tuple[dict[str, Any], str | None]] = []
    for message in messages:
        if message.get("role") == "assistant":
            payload = parse_json(message.get("content"))
            if payload is not None:
                pairs.append((payload, None))
        elif message.get("role") == "tool" and pairs:
            body = parse_json(message.get("content")) or {}
            result = body.get("result")
            if isinstance(result, dict):
                call, _ = pairs[-1]
                pairs[-1] = (call, str(result.get("status")))
    return pairs


def validate_messages(row: dict[str, Any]) -> tuple[bool, str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return False, "missing_messages"
    if not any(item.get("role") == "user" for item in messages):
        return False, "missing_user_turn"
    assistant_calls: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        payload = parse_json(message.get("content"))
        if payload is None or not isinstance(payload.get("args"), dict):
            return False, "malformed_assistant_json"
        if payload.get("tool") not in ALLOWED_TOOLS:
            return False, "unknown_tool"
        assistant_calls.append(payload)
        if payload["tool"] == "get_historical_prices":
            period = payload["args"].get("period")
            if period is not None and period not in PERIODS:
                return False, "invalid_period"
    if len(assistant_calls) > 6:
        return False, "trajectory_too_long"
    if not assistant_calls or assistant_calls[-1]["tool"] != "finish":
        return False, "does_not_finish"
    # One repeated call is a legitimate bounded retry when the previous result
    # was unusable (the loop_trap and malformed_error intents are built on it).
    # A repeat after a usable result, or more than one retry, is a defect.
    attempts: dict[str, list[str | None]] = {}
    for call, status in call_sequence(messages):
        if call["tool"] == "finish":
            continue
        key = json.dumps(call, sort_keys=True)
        history = attempts.setdefault(key, [])
        if history:
            if len(history) >= MAX_ATTEMPTS_PER_CALL:
                return False, "too_many_retries"
            if history[-1] in USABLE_STATUSES:
                return False, "duplicate_call"
        history.append(status)
    for message in messages:
        if message.get("role") == "assistant" and ADVICE_WORDS.search(
            message.get("content", "")
        ):
            return False, "unsafe_advice_text"
    if len(messages) < 3:
        return False, "trajectory_too_short"
    return True, "accepted"


def load_gold() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for line in INPUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index[row["question"]] = row
    return index


def run_curatorkit(max_samples: int | None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = CuratorConfig(
        dataset=str(INPUT),
        format="auto",
        max_samples=max_samples,
        min_tokens=0,
        dedup=DEDUP,
        minhash_threshold=MINHASH_THRESHOLD,
        clean=True,
        export_formats=["messages"],
        output_dir=str(OUTPUT_DIR),
        name="finance_agent_ws4",
        write_hf_readme=True,
    )
    Curator(config).run()


def post_validate(gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not CURATORKIT_OUTPUT.exists():
        raise FileNotFoundError(f"CuratorKIT did not write {CURATORKIT_OUTPUT}")
    counts: Counter[str] = Counter()
    per_intent: Counter[str] = Counter()
    accepted_messages: list[str] = []
    curated: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for line in CURATORKIT_OUTPUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ok, reason = validate_messages(row)
        counts[reason] += 1
        if not ok:
            continue
        accepted_messages.append(json.dumps(row, sort_keys=True))
        question = row["messages"][1]["content"]
        source = gold.get(question)
        if source is None:
            unmatched.append(question)
            continue
        record = {key: source[key] for key in METADATA_KEYS}
        record["messages"] = row["messages"]
        curated.append(record)
        per_intent[source["intent"]] += 1

    if unmatched:
        raise RuntimeError(
            f"{len(unmatched)} exported rows do not match a gold question, "
            f"e.g. {unmatched[0]!r}"
        )
    if not curated:
        raise RuntimeError(f"no rows survived the project gates: {dict(counts)}")

    curated.sort(key=lambda item: item["id"])
    VALIDATED.write_text(
        "\n".join(accepted_messages) + ("\n" if accepted_messages else ""),
        encoding="utf-8",
    )
    CURATED.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in curated) + "\n",
        encoding="utf-8",
    )
    GATE_LOG.write_text(
        json.dumps(dict(sorted(counts.items())), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    gold_total = Counter(row["intent"] for row in gold.values())
    summary = {
        "curatorkit_settings": {
            "min_tokens": 0,
            "dedup": DEDUP,
            "minhash_threshold": MINHASH_THRESHOLD,
            "minhash_ngram": 3,
            "format": "auto",
        },
        "counts": {
            "gold_input": len(gold),
            "curatorkit_exported": sum(counts.values()),
            "project_gates_accepted": len(curated),
            "project_gate_reasons": dict(sorted(counts.items())),
        },
        "per_intent": {
            intent: {"gold": gold_total[intent], "curated": per_intent[intent]}
            for intent in sorted(gold_total)
        },
        "sha256": {path.name: sha256(path) for path in (CURATED, VALIDATED)},
    }
    MANIFEST.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    CHECKSUMS.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (CURATED, VALIDATED, CURATORKIT_OUTPUT, GATE_LOG, MANIFEST)
        ),
        encoding="utf-8",
    )
    fix_readme_hash()
    return summary


def fix_readme_hash() -> None:
    """CuratorKIT's README quotes a hash that is not the file it names."""
    readme = OUTPUT_DIR / "README.md"
    if not readme.exists():
        return
    digest = sha256(CURATORKIT_OUTPUT)
    lines = readme.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("description:"):
            lines[index] = f"  description: curatorkit messages export, sha256 {digest}"
            break
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if not INPUT.exists():
        raise SystemExit(f"missing input: {INPUT}")
    run_curatorkit(args.max_samples)
    summary = post_validate(load_gold())
    print(json.dumps(summary["counts"], sort_keys=True))
    print(json.dumps(summary["per_intent"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
