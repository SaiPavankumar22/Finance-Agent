"""Add training coverage for the three India families that failed the gates.

The rejected candidate got every Indian quote, ratio, history and refusal row
right, and missed all four news rows plus every invalid-symbol and
unsupported-action row. Two different mistakes:

* news: the model answered with a quote-style call instead of
  ``get_company_news``, even though the recorded Indian news fixture is
  unavailable - the correct behaviour is still to ask the news tool first;
* invalid symbols and unsupported actions: the model called a tool instead of
  declining, so the row must teach ``finish`` directly.

Questions are worded so they never collide with ``eval/india_set.jsonl``, which
stays unseen.
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
FIXTURES = PROJECT_ROOT / "data" / "fixtures"
MANIFEST = PROJECT_ROOT / "data" / "splits" / "india_coverage_manifest.json"
HELDOUT = (
    PROJECT_ROOT / "data" / "splits" / "val.jsonl",
    PROJECT_ROOT / "eval" / "testset.jsonl",
    PROJECT_ROOT / "eval" / "safety_set.jsonl",
    PROJECT_ROOT / "eval" / "india_set.jsonl",
)
STAGE = "india-coverage-v2"
#: Weight 1: at weight 2 the mixture diluted planning to 0.7333 frozen.
WEIGHT = 1
FINISH = {"tool": "finish", "args": {}}

NEWS_TICKERS = ("RELIANCE.NS", "TCS.NS", "INFY.NS")
NEWS_PHRASINGS = (
    "Summarise the news flow on {ticker}.",
    "What has the press written about {ticker} lately?",
    "Give me the news tape for {ticker}.",
    "Any press coverage on {ticker} in the last few days?",
)
INVALID_TICKERS = ("NIFTY50.NS", "NIFTYBANK.NS", "SENSEX.NS", "NIFTY50.BO")
INVALID_PHRASINGS = (
    "Quote check on {ticker}.",
    "Price for {ticker} right now.",
    "Pull up {ticker} for me.",
    "What is {ticker} trading at?",
)
UNSUPPORTED = (
    ("Dividend history for {ticker}.", "RELIANCE.NS"),
    ("Split schedule for {ticker}.", "TCS.NS"),
    ("Bonus issue history for {ticker}.", "INFY.NS"),
    ("Buyback details for {ticker}.", "RELIANCE.NS"),
    ("Dividend payouts for {ticker} please.", "TCS.NS"),
    ("NSE trading hours please.", "RELIANCE.NS"),
    ("When do the Indian markets close?", "TCS.NS"),
    ("Which index does {ticker} belong to?", "INFY.NS"),
    ("Circuit limit for {ticker}.", "RELIANCE.NS"),
    ("Upcoming IPO calendar for the NSE.", "TCS.NS"),
)


def load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def dump(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def news_result(ticker: str) -> dict[str, Any]:
    """The recorded Indian news fixture, unchanged.

    Real NSE news is unavailable, which is why the gold trajectory asks the news
    tool and then finishes rather than inventing headlines.
    """
    path = FIXTURES / f"{ticker}_get_company_news.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("result", payload)


def messages_for(
    question: str, calls: list[tuple[str, dict[str, Any], dict[str, Any] | None]]
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for index, (tool, args, result) in enumerate(calls, start=1):
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps({"tool": tool, "args": args}, separators=(",", ":")),
            }
        )
        if result is not None:
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        {"call_id": f"c{index}", "result": result},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
    return messages


def main() -> int:
    rows = [row for row in load(TRAIN) if row.get("stage") not in {STAGE, "india-coverage-v1"}]
    train_questions = {row["question"] for row in rows}
    heldout = set()
    for path in HELDOUT:
        heldout |= {row["question"] for row in load(path)}

    new_rows: list[dict[str, Any]] = []
    counter = 0

    for ticker, phrasing in (
        (ticker, phrasing) for phrasing in NEWS_PHRASINGS for ticker in NEWS_TICKERS
    ):
        counter += 1
        question = phrasing.format(ticker=ticker)
        news = news_result(ticker)
        args = {"ticker": ticker, "limit": 5}
        new_rows.append(
            {
                "id": f"ind_{counter:04d}",
                "question": question,
                "tickers": [ticker],
                "intent": "india_news",
                "expected_behavior": "report_news_unavailable",
                "weight": WEIGHT,
                "stage": STAGE,
                "messages": messages_for(
                    question,
                    [("get_company_news", args, news), ("finish", {}, None)],
                ),
            }
        )

    for phrasing in INVALID_PHRASINGS:
        for ticker in INVALID_TICKERS:
            counter += 1
            question = phrasing.format(ticker=ticker)
            new_rows.append(
                {
                    "id": f"ind_{counter:04d}",
                    "question": question,
                    "tickers": [ticker],
                    "intent": "invalid_symbol",
                    "expected_behavior": "abstain_invalid_ticker",
                    "weight": WEIGHT,
                    "stage": STAGE,
                    "messages": messages_for(question, [("finish", {}, None)]),
                }
            )

    for question, ticker in UNSUPPORTED:
        counter += 1
        text = question.format(ticker=ticker) if "{ticker}" in question else question
        new_rows.append(
            {
                "id": f"ind_{counter:04d}",
                "question": text,
                "tickers": [ticker],
                "intent": "unsupported_india",
                "expected_behavior": "say_unsupported",
                "weight": WEIGHT,
                "stage": STAGE,
                "messages": messages_for(text, [("finish", {}, None)]),
            }
        )

    collisions = [row["id"] for row in new_rows if row["question"] in train_questions | heldout]
    assert not collisions, f"questions collide with existing data: {collisions}"
    assert len({row["question"] for row in new_rows}) == len(new_rows), "duplicate questions"

    combined = rows + new_rows
    TRAIN.write_text(dump(combined), encoding="utf-8")
    manifest = {
        "stage": STAGE,
        "added": len(new_rows),
        "rows": len(combined),
        "weighted_examples": sum(row["weight"] for row in combined),
        "news_rows": sum(1 for row in new_rows if row["intent"] == "india_news"),
        "invalid_rows": sum(1 for row in new_rows if row["intent"] == "invalid_symbol"),
        "unsupported_rows": sum(1 for row in new_rows if row["intent"] == "unsupported_india"),
        "sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest()[:12],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
