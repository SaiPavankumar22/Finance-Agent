"""Run CuratorKIT and project-specific gates over the WS4 gold set.

CuratorKIT performs the real schema, deduplication, and cleaning stages. The
post-checks below enforce this project's trajectory contract because generic
dataset curation cannot know the finance tool registry.

Usage:
    python pipelines/06_curate.py
    python pipelines/06_curate.py --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from curatorkit import Curator, CuratorConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "data" / "gold" / "trajectories.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "curated"
VALIDATED = OUTPUT_DIR / "validated_messages.jsonl"
GATE_LOG = OUTPUT_DIR / "gate_counts.json"
ALLOWED_TOOLS = {
    "get_stock_quote",
    "get_financial_ratios",
    "get_historical_prices",
    "get_company_news",
    "finish",
}
PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y"}
ADVICE_WORDS = re.compile(r"\b(buy|sell|guaranteed|will double)\b", re.I)


def parse_json(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


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
    seen = set()
    for call in assistant_calls[:-1]:
        key = json.dumps(call, sort_keys=True)
        if key in seen and call["tool"] != "finish":
            return False, "duplicate_call"
        seen.add(key)
    for message in messages:
        if message.get("role") == "assistant" and ADVICE_WORDS.search(
            message.get("content", "")
        ):
            return False, "unsafe_advice_text"
    return True, "accepted"


def run_curatorkit(max_samples: int | None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = CuratorConfig(
        dataset=str(INPUT),
        format="auto",
        max_samples=max_samples,
        dedup="minhash",
        minhash_threshold=0.85,
        clean=True,
        export_formats=["messages"],
        output_dir=str(OUTPUT_DIR),
        name="finance_agent_ws4",
        write_hf_readme=True,
    )
    Curator(config).run()


def post_validate() -> dict[str, int]:
    source = OUTPUT_DIR / "sft_messages.jsonl"
    if not source.exists():
        raise FileNotFoundError(f"CuratorKIT did not write {source}")
    counts: Counter[str] = Counter()
    accepted: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ok, reason = validate_messages(row)
        counts[reason] += 1
        if ok:
            accepted.append(json.dumps(row, sort_keys=True))
    if counts["accepted"] and counts["accepted"] < sum(counts.values()) / 2:
        raise RuntimeError(f"project gates rejected more than half: {dict(counts)}")
    VALIDATED.write_text(
        "\n".join(accepted) + ("\n" if accepted else ""),
        encoding="utf-8",
    )
    result = dict(sorted(counts.items()))
    GATE_LOG.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if not INPUT.exists():
        raise SystemExit(f"missing input: {INPUT}")
    run_curatorkit(args.max_samples)
    print(json.dumps(post_validate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
