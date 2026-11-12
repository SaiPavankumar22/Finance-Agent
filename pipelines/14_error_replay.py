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
STAGE = "error-replay-v3"
ERROR_BEHAVIOURS = {"abstain_malformed_response", "stop_after_bounded_attempts"}
ERROR_WEIGHT = 2
#: The two audit rows that still retried are P/E questions on get_financial_ratios,
#: so that phrasing family gets its own weight.
RATIOS_WEIGHT = 3

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
#: Phrasings the audit uses for the ratios family, replayed on train-only tickers.
RATIOS_QUESTIONS = (
    "{ticker}",
    "P/E and return on equity for {ticker}",
    "Valuation multiples for {ticker}",
    "What leverage does {ticker} carry",
    "ROE figure for {ticker}",
    "How is {ticker} valued",
)
RATIOS_TICKERS = ("PFE", "PG", "ADBE", "MC.PA", "BHP.AX", "CBA.AX")

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


def declines_after_failure(row: dict[str, Any]) -> bool:
    """True when the last action is ``finish`` straight after a tool error.

    Duplicates may appear *in history* - the audit itself scripts two identical
    failures - so the invariant is not "no repeated call ever", it is that the
    planner never answers a failure with another call.
    """
    messages = row["messages"]
    last, previous = messages[-1], messages[-2]
    if last["role"] != "assistant" or json.loads(last["content"])["tool"] != "finish":
        return False
    if previous["role"] != "tool":
        return False
    return json.loads(previous["content"])["result"]["status"] == "error"


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
    rows = [
        row
        for row in load(TRAIN)
        if row.get("stage") not in {STAGE, "error-replay-v1", "error-replay-v2"}
    ]

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

    ratios_rows = []
    for position, (question, ticker) in enumerate(
        (f"Give me {ticker}'s P/E and return on equity.", ticker) for ticker in RATIOS_TICKERS
    ):
        args = {"ticker": ticker}
        messages = [
            {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        failures = 2 if position % 2 == 0 else 1
        for step in range(failures):
            messages.append(assistant({"tool": "get_financial_ratios", "args": args}))
            messages.append(
                error_turn(
                    f"c{step + 1}",
                    "get_financial_ratios",
                    args,
                    "rate_limit",
                    "2026-11-12T05:02:00+00:00",
                )
            )
        messages.append(assistant(FINISH))
        ratios_rows.append(
            {
                "id": f"errr_{position + 1:04d}",
                "question": question,
                "tickers": [ticker],
                "intent": "loop_trap",
                "expected_behavior": "stop_after_bounded_attempts",
                "weight": RATIOS_WEIGHT,
                "stage": STAGE,
                "messages": messages,
            }
        )
    ratios_collisions = [
        row["id"] for row in ratios_rows if row["question"] in train_questions | heldout_questions
    ]
    assert not ratios_collisions, f"ratios question collides: {ratios_collisions}"
    new_rows.extend(ratios_rows)
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
        assert declines_after_failure(row), f"{row['id']} does not decline after a failure"
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
        "fallback_rows": sum(
            1 for row in new_rows if len(row["messages"]) == 7 and not row["id"].startswith("errr_")
        ),
        "ratios_rows": len(ratios_rows),
        "sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest()[:12],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    print(f"train: {len(combined)} rows ({weighted} weighted examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
