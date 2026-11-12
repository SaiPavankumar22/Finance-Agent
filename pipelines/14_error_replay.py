"""Weight the error-replay rows and add explicit retry-then-finish trajectories.

The rejected prompt-conditioned candidate failed exactly one gate: on 2 of 10
audit rows the model re-issued a call that had already failed, instead of
finishing. The training set already contains 51 such rows, so the gap is not
absence of examples - it is that they are 6.8% of the data and all share a few
phrasings. This stage does two things:

* tags every row with an integer ``weight`` and gives the error-replay families
  (``abstain_malformed_response``, ``stop_after_bounded_attempts``) weight 3, so
  two failures followed by ``finish`` is seen far more often per epoch;
* adds 24 fresh retry-then-finish trajectories over tools and tickers the audit
  uses, including Indian symbols and all three recorded error kinds
  (``insufficient_data``, ``rate_limit``, ``malformed_response``).

Every new row carries the WS6 prompt verbatim and is checked against the
validation, frozen test, safety and India sets.
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
STAGE = "error-replay-v1"

ERROR_WEIGHT = 3
NEW_WEIGHT = 3

#: ``(question, ticker, tool, error)`` - questions never reuse a held-out string.
CASES: tuple[tuple[str, str, str, str], ...] = (
    ("Give me the full picture on 7203.T.", "7203.T", "get_stock_quote", "insufficient_data"),
    ("Pull the complete report for 6758.T.", "6758.T", "get_financial_ratios", "rate_limit"),
    (
        "I need everything you have on 005930.KS.",
        "005930.KS",
        "get_historical_prices",
        "malformed_response",
    ),
    ("Run a full workup on ADBE.", "ADBE", "get_company_news", "insufficient_data"),
    ("Give me the full picture on ASML.AS.", "ASML.AS", "get_stock_quote", "rate_limit"),
    (
        "Pull the complete report for BHP.AX.",
        "BHP.AX",
        "get_financial_ratios",
        "malformed_response",
    ),
    (
        "I need everything you have on CBA.AX.",
        "CBA.AX",
        "get_historical_prices",
        "insufficient_data",
    ),
    ("Run a full workup on MC.PA.", "MC.PA", "get_company_news", "rate_limit"),
    ("Give me the full picture on PFE.", "PFE", "get_stock_quote", "malformed_response"),
    ("Pull the complete report for PG.", "PG", "get_financial_ratios", "insufficient_data"),
    ("I need everything you have on SBIN.NS.", "SBIN.NS", "get_historical_prices", "rate_limit"),
    ("Run a full workup on HDFCBANK.NS.", "HDFCBANK.NS", "get_company_news", "malformed_response"),
    ("Pull the complete report for 7203.T.", "7203.T", "get_stock_quote", "insufficient_data"),
    ("I need everything you have on 6758.T.", "6758.T", "get_financial_ratios", "rate_limit"),
    ("Run a full workup on 005930.KS.", "005930.KS", "get_historical_prices", "malformed_response"),
    ("Give me the full picture on ADBE.", "ADBE", "get_company_news", "insufficient_data"),
    ("Pull the complete report for ASML.AS.", "ASML.AS", "get_stock_quote", "rate_limit"),
    (
        "I need everything you have on BHP.AX.",
        "BHP.AX",
        "get_financial_ratios",
        "malformed_response",
    ),
    ("Run a full workup on CBA.AX.", "CBA.AX", "get_historical_prices", "insufficient_data"),
    ("Give me the full picture on MC.PA.", "MC.PA", "get_company_news", "rate_limit"),
    ("Pull the complete report for PFE.", "PFE", "get_stock_quote", "malformed_response"),
    ("I need everything you have on PG.", "PG", "get_financial_ratios", "insufficient_data"),
    ("Run a full workup on SBIN.NS.", "SBIN.NS", "get_historical_prices", "rate_limit"),
    (
        "Give me the full picture on HDFCBANK.NS.",
        "HDFCBANK.NS",
        "get_company_news",
        "malformed_response",
    ),
)

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


def error_result(tool: str, args: dict[str, Any], error: str, retrieved_at: str) -> dict[str, Any]:
    return {
        "tool": tool,
        "args": args,
        "status": "error",
        "error": error,
        "data": None,
        "source": "yfinance" if tool != "get_company_news" else "news",
        "retrieved_at": retrieved_at,
        "as_of": None,
    }


def retry_then_finish(
    index: int, question: str, ticker: str, tool: str, error: str
) -> dict[str, Any]:
    args = ARGS[tool](ticker)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for call_id, retrieved_at in (
        ("c1", "2026-11-11T09:31:00+00:00"),
        ("c2", "2026-11-11T09:31:04+00:00"),
    ):
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps({"tool": tool, "args": args}, separators=(",", ":")),
            }
        )
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(
                    {"call_id": call_id, "result": error_result(tool, args, error, retrieved_at)},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    messages.append({"role": "assistant", "content": json.dumps(FINISH)})
    return {
        "id": f"err_{index:04d}",
        "question": question,
        "tickers": [ticker],
        "intent": "loop_trap",
        "expected_behavior": "stop_after_bounded_attempts",
        "weight": NEW_WEIGHT,
        "stage": STAGE,
        "messages": messages,
    }


def main() -> int:
    rows = load(TRAIN)
    rows = [row for row in rows if row.get("stage") != STAGE]

    train_questions = {row["question"] for row in rows}
    heldout_questions: set[str] = set()
    for path in HELDOUT:
        heldout_questions |= {row["question"] for row in load(path)}

    new_rows = [
        retry_then_finish(index, question, ticker, tool, error)
        for index, (question, ticker, tool, error) in enumerate(CASES, start=1)
    ]
    collisions = [
        row["id"] for row in new_rows if row["question"] in train_questions | heldout_questions
    ]
    assert not collisions, f"question collides with existing data: {collisions}"
    assert len({row["question"] for row in new_rows}) == len(new_rows)

    for row in rows:
        row["weight"] = (
            ERROR_WEIGHT
            if row["expected_behavior"]
            in {
                "abstain_malformed_response",
                "stop_after_bounded_attempts",
            }
            else 1
        )

    combined = rows + new_rows
    assert all(isinstance(row["weight"], int) and row["weight"] >= 1 for row in combined)
    weighted = sum(row["weight"] for row in combined)
    error_rows = sum(
        1 for row in combined if row["expected_behavior"] == "stop_after_bounded_attempts"
    )

    TRAIN.write_text(dump(combined), encoding="utf-8")
    manifest = {
        "stage": STAGE,
        "rows": len(combined),
        "added": len(new_rows),
        "weighted_examples": weighted,
        "error_replay_rows": error_rows,
        "error_weight": ERROR_WEIGHT,
        "error_share_of_examples": round(
            sum(
                row["weight"]
                for row in combined
                if row["expected_behavior"] == "stop_after_bounded_attempts"
            )
            / weighted,
            4,
        ),
        "kinds": sorted({case[3] for case in CASES}),
        "sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest()[:12],
        "overlap_with_heldout": {
            path.name: sorted(
                {row["question"] for row in new_rows} & {r["question"] for r in load(path)}
            )
            for path in HELDOUT
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    print(f"train: {len(rows)} -> {len(combined)} rows ({weighted} weighted examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
