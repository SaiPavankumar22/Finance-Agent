"""Enrich deterministic WS4 trajectories with tightly constrained Anthropic text.

The model is used only for linguistic variation, never for tool selection or
financial values. Every response is cached and rejected unless it preserves the
ticker and passes the placeholder/safety checks.

Usage:
    python pipelines/05_llm_enrich.py --limit 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "data" / "gold" / "trajectories.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "enriched"
CACHE_DIR = OUTPUT_DIR / "cache"
OUTPUT = OUTPUT_DIR / "trajectories.jsonl"
MODEL_DEFAULT = "claude-haiku-5-5"
PLACEHOLDER = re.compile(r"\{\{E\d+\}\}")
ADVICE_WORDS = re.compile(r"\b(buy|sell|guaranteed|must invest|will double)\b", re.I)


def load_rows(limit: int) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in INPUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def cache_path(model: str, row: dict[str, Any]) -> Path:
    key = hashlib.sha256(
        f"ws4-enrich-v2:{model}:{row['id']}:{row['question']}".encode()
    ).hexdigest()
    return CACHE_DIR / f"{key}.json"


def json_text(response: Any) -> str:
    return "".join(
        block.text for block in response.content if hasattr(block, "text")
    )


def parse_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("response is not an object")
    variants = payload.get("question_variants")
    interpretations = payload.get("interpretations")
    if not isinstance(variants, list) or not isinstance(interpretations, list):
        raise ValueError("missing question_variants or interpretations arrays")
    if not all(isinstance(item, str) and item.strip() for item in variants):
        raise ValueError("question variants must be non-empty strings")
    if not all(isinstance(item, str) and item.strip() for item in interpretations):
        raise ValueError("interpretations must be non-empty strings")
    return {
        "question_variants": [item.strip() for item in variants[:2]],
        "interpretations": [item.strip() for item in interpretations[:2]],
    }


def validate_payload(
    row: dict[str, Any], payload: dict[str, Any]
) -> tuple[bool, str | None]:
    question = row["question"]
    for variant in payload["question_variants"]:
        for ticker in row["tickers"]:
            if ticker not in variant:
                return False, f"variant dropped ticker {ticker}"
    for interpretation in payload["interpretations"]:
        if re.search(r"\d", PLACEHOLDER.sub("", interpretation)):
            return False, "interpretation contains an ungrounded digit"
        if ADVICE_WORDS.search(interpretation):
            return False, "interpretation contains advice language"
        placeholders = {
            item[2:-2] for item in PLACEHOLDER.findall(interpretation)
        }
        ledger_ids = {
            entry["id"] for entry in row["report"].get("ledger", [])
        }
        if not placeholders.issubset(ledger_ids):
            return False, "interpretation references unknown evidence"
    if not question:
        return False, "empty source question"
    return True, None


def prompt_for(row: dict[str, Any]) -> str:
    return f"""You are enriching a verified financial-agent training example.
Return JSON only with exactly two arrays:
{{"question_variants":["..."],"interpretations":["..."]}}

Rules:
- Write two concise English variants of the user question.
- Preserve every ticker exactly: {", ".join(row["tickers"])}.
- Preserve the request's intent and time period. Do not add facts.
- Write one or two cautious interpretation sentences using only evidence
  placeholders from the ledger. Use placeholders such as {{{{E1}}}} exactly.
- Do not write any digits except digits inside evidence placeholders.
- Do not recommend buying or selling, predict returns, or claim certainty.
- Do not invent prices, ratios, dates, companies, or headlines.

Intent: {row["intent"]}
Original question: {row["question"]}
Evidence ledger:
{row["report"].get("ledger_text", "")}
"""


def enrich_row(client: Any, model: str, row: dict[str, Any]) -> dict[str, Any]:
    path = cache_path(model, row)
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        raw = cached["raw"]
        usage = cached.get("usage", {})
    else:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            thinking={"type": "disabled"},
            system="You output strict JSON and follow the requested safety rules.",
            messages=[{"role": "user", "content": prompt_for(row)}],
        )
        raw = json_text(response)
        usage_obj = getattr(response, "usage", None)
        usage = {
            "input_tokens": getattr(usage_obj, "input_tokens", None),
            "output_tokens": getattr(usage_obj, "output_tokens", None),
        }
        path.write_text(
            json.dumps({"raw": raw, "usage": usage}, sort_keys=True),
            encoding="utf-8",
        )

    try:
        payload = parse_payload(raw)
        valid, reason = validate_payload(row, payload)
    except (ValueError, json.JSONDecodeError) as error:
        payload = {}
        valid, reason = False, str(error)
    return {
        "id": row["id"],
        "source_question": row["question"],
        "intent": row["intent"],
        "mutation": row["mutation"],
        "status": "accepted" if valid else "rejected",
        "rejection_reason": reason,
        "question_variants": payload.get("question_variants", []),
        "interpretations": payload.get("interpretations", []),
        "usage": usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    load_dotenv(override=False)
    model = args.model or os.getenv("CLAUDE_MODEL", MODEL_DEFAULT)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    if not INPUT.exists():
        raise SystemExit(f"missing input: {INPUT}")

    from anthropic import Anthropic

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = Anthropic()
    enriched = [
        enrich_row(client, model, row)
        for row in load_rows(args.limit)
    ]
    OUTPUT.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in enriched) + "\n",
        encoding="utf-8",
    )
    counts = {
        "accepted": sum(item["status"] == "accepted" for item in enriched),
        "rejected": sum(item["status"] == "rejected" for item in enriched),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "model": model,
                "requested": len(enriched),
                "counts": counts,
                "cache_dir": str(CACHE_DIR.relative_to(PROJECT_ROOT)),
                "policy": (
                    "LLM text only; deterministic tool calls and values remain authoritative."
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
