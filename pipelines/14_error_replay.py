"""Make the error-replay rows match the rule they are scored by.

The audit's post-error rule is ``declined or (fallback and not repeated)``: after
a provider failure the planner may decline, or switch to a *different* tool, but
re-issuing the failed call is counted as a repeated failing call. The existing
training rows did exactly that - call, error, **the same call again**, error,
finish - so weighting them more (the previous attempt) changed nothing, because
the data taught the behaviour the gate penalises.

This stage:

* rewrites those rows to a single failure followed by ``finish``;
* adds 24 fresh trajectories, half of them one-failure-then-finish, half a
  fallback to a different tool before finishing;
* weights the error families 2 (down from 3, which cost planning accuracy);
* asserts that no gold error trajectory contains an identical retry.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.safety import SAFETY_SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN = PROJECT_ROOT / "data" / "splits" / "train.jsonl"
HELDOUT = (
    PROJECT_ROOT / "data" / "splits" / "val.jsonl",
    PROJECT_ROOT / "eval" / "testset.jsonl",
    PROJECT_ROOT / "eval" / "safety_set.jsonl",
    PROJECT_ROOT / "eval" / "india_set.jsonl",
)
MANIFEST = PROJECT_ROOT / "data" / "splits" / "error_replay_manifest.json"
STAGE = "error-replay-v2"
ERROR_BEHAVIOURS = {"abstain_malformed_response", "stop_after_bounded_attempts"}
ERROR_WEIGHT = 2

#: Tickers that appear only in training, plus the tools and errors to replay.
TICKERS = (
    "7203.T",
    "6758.T",
    "005930.KS",
    "ADBE",
    "ASML.AS",
    "BHP.AX",
    "CBA.AX",
    "MC.PA",
    "PFE",
    "PG",
    "SBIN.NS",
    "HDFCBANK.NS",
)
TOOLS = ("get_stock_quote", "get_financial_ratios", "get_historical_prices", "get_company_news")
ERRORS = ("insufficient_data", "rate_limit", "malformed_response")
PHRASINGS = (
    "Give me the full picture on {ticker}.",
    "Pull the complete report for {ticker}.",
    "I need everything you have on {ticker}.",
    "Run a full workup on {ticker}.",
)
#: A different tool that can legitimately answer a similar question.
FALLBACK = {
    "get_stock_quote": "get_historical_prices",
    "get_financial_ratios": "get_stock_quote",
    "get_historical_prices": "get_stock_quote",
    "get_company_news": "get_stock_quote",
}
ARGS = {
    "get_stock_quote": lambda ticker: {"ticker": ticker},
    "get_financial_ratios": lambda ticker: {"ticker": ticker},
    "get_historical_prices": lambda ticker: {"ticker": ticker, "period": "1y"},
    "get_company_news": lambda ticker: {"ticker": ticker, "limit": 5},
}
FINISH = {"tool": "finish", "args": {}}


def load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def dump(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def call(tool: str, ticker: str) -> dict[str, Any]:
    return {"tool": tool, "args": ARGS[tool](ticker)}


def error_turn(
    call_id: str, tool: str, args: dict[str, Any], error: str, at: str
) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": json.dumps(
            {
                "call_id": call_id,
                "result": {
                    "tool": tool,
                    "args": args,
                    "status": "error",
                    "error": error,
                    "data": None,
                    "source": "news" if tool == "get_company_news" else "yfinance",
                    "retrieved_at": at,
                    "as_of": None,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def assistant(payload: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": json.dumps(payload, separators=(",", ":"))}


def new_row(
    index: int, question: str, ticker: str, tool: str, error: str, fallback: bool
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    args = ARGS[tool](ticker)
    messages.append(assistant(call(tool, ticker)))
    messages.append(error_turn("c1", tool, args, error, "2026-11-12T04:31:00+00:00"))
    if fallback:
        other = FALLBACK[tool]
        other_args = ARGS[other](ticker)
        messages.append(assistant(call(other, ticker)))
        messages.append(error_turn("c2", other, other_args, error, "2026-11-12T04:31:05+00:00"))
    messages.append(assistant(FINISH))
    return {
        "id": f"err_{index:04d}",
        "question": question,
        "tickers": [ticker],
        "intent": "loop_trap",
        "expected_behavior": "stop_after_bounded_attempts",
        "weight": ERROR_WEIGHT,
        "stage": STAGE,
        "messages": messages,
    }


def distinct_retries(row: dict[str, Any]) -> bool:
    """True when no assistant turn repeats an earlier failed call."""
    payloads = [
        json.dumps(json.loads(message["content"]), sort_keys=True)
        for message in row["messages"]
        if message["role"] == "assistant"
    ]
    return len(payloads) == len(set(payloads))


def rewrite_existing(row: dict[str, Any]) -> dict[str, Any]:
    """Replace an identical retry with an immediate decline.

    ``call, error, same call, error, finish`` teaches the planner that repeating
    a failed call is the path to a decline. The scored rule says the opposite, so
    the retry is dropped and the trajectory becomes ``call, error, finish``.
    """
    messages = row["messages"]
    assistant_positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if len(assistant_positions) < 3:
        return row
    first, second, last = assistant_positions[0], assistant_positions[1], assistant_positions[-1]
    if json.loads(messages[first]["content"]) != json.loads(messages[second]["content"]):
        return row
    kept = messages[:second]
    kept.append(messages[last])
    row["messages"] = kept
    row["rewritten_by"] = STAGE
    return row


def main() -> int:
    rows = [row for row in load(TRAIN) if row.get("stage") not in {STAGE, "error-replay-v1"}]

    train_questions = {row["question"] for row in rows}
    heldout_questions: set[str] = set()
    heldout_tickers: set[str] = set()
    for path in HELDOUT:
        heldout_rows = load(path)
        heldout_questions |= {row["question"] for row in heldout_rows}
        heldout_tickers |= {ticker for row in heldout_rows for ticker in row.get("tickers") or []}

    new_rows = []
    index = 0
    for round_index in range(2):
        for ticker in TICKERS:
            tool = TOOLS[index % len(TOOLS)]
            error = ERRORS[(index + round_index) % len(ERRORS)]
            question = PHRASINGS[(index + round_index) % len(PHRASINGS)].format(ticker=ticker)
            new_rows.append(
                new_row(index + 1, question, ticker, tool, error, fallback=index % 2 == 1)
            )
            index += 1

    collisions = [
        row["id"] for row in new_rows if row["question"] in train_questions | heldout_questions
    ]
    assert not collisions, f"question collides with existing data: {collisions}"
    assert len({row["question"] for row in new_rows}) == len(new_rows)
    leaked = sorted({row["tickers"][0] for row in new_rows} & heldout_tickers)
    assert not leaked, f"tickers appear in a held-out set: {leaked}"

    rewritten = 0
    for row in rows:
        if row["expected_behavior"] in ERROR_BEHAVIOURS:
            before = row["messages"]
            row = rewrite_existing(row)
            rewritten += int(row["messages"] != before)
        row["weight"] = ERROR_WEIGHT if row["expected_behavior"] in ERROR_BEHAVIOURS else 1

    combined = rows + new_rows
    error_rows = [
        row for row in combined if row["expected_behavior"] == "stop_after_bounded_attempts"
    ]
    for row in error_rows:
        assert distinct_retries(row), f"{row['id']} repeats a failed call"
        assert json.loads(row["messages"][-1]["content"])["tool"] == "finish", row["id"]

    weighted = sum(row["weight"] for row in combined)
    TRAIN.write_text(dump(combined), encoding="utf-8")
    manifest = {
        "stage": STAGE,
        "rows": len(combined),
        "added": len(new_rows),
        "rewritten_trajectories": rewritten,
        "weighted_examples": weighted,
        "error_replay_rows": len(error_rows),
        "error_weight": ERROR_WEIGHT,
        "error_share_of_examples": round(sum(row["weight"] for row in error_rows) / weighted, 4),
        "fallback_rows": sum(1 for row in new_rows if len(row["messages"]) == 7),
        "sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest()[:12],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    print(f"train: {len(combined)} rows ({weighted} weighted examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
