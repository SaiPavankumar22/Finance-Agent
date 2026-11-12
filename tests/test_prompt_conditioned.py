"""Prompt-conditioned training data and the India-only held-out set."""

import json
from pathlib import Path

from eval.metrics import Prediction, score_prediction
from eval.safety import SAFETY_SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN = PROJECT_ROOT / "data" / "splits" / "train.jsonl"
VAL = PROJECT_ROOT / "data" / "splits" / "val.jsonl"
TESTSET = PROJECT_ROOT / "eval" / "testset.jsonl"
SAFETY = PROJECT_ROOT / "eval" / "safety_set.jsonl"
INDIA = PROJECT_ROOT / "eval" / "india_set.jsonl"

ABSTAIN = {
    "refuse_advice",
    "abstain_invalid_ticker",
    "abstain_malformed_response",
    "stop_after_bounded_attempts",
    "say_unsupported",
}


def load(path):
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_every_training_row_uses_the_ws6_prompt():
    rows = load(TRAIN)
    assert rows
    assert all(row["messages"][0]["content"] == SAFETY_SYSTEM_PROMPT for row in rows)


def test_refusal_rows_end_in_finish_and_benign_rows_start_with_a_tool():
    rows = load(TRAIN)
    for row in rows:
        if row["expected_behavior"] in ABSTAIN:
            assert json.loads(row["messages"][-1]["content"])["tool"] == "finish", row["id"]
        else:
            first = json.loads(row["messages"][2]["content"])
            assert first["tool"] != "finish", f"{row['id']} is benign but declines"
            ticker = first.get("args", {}).get("ticker")
            assert ticker is None or ticker in row["tickers"], row["id"]


def test_training_rows_stay_disjoint_from_every_heldout_set():
    train = {row["question"] for row in load(TRAIN)}
    for path in (VAL, TESTSET, SAFETY, INDIA):
        overlap = {row["question"] for row in load(path)} & train
        assert not overlap, f"{path.name} leaks into training: {sorted(overlap)[:3]}"

    val_tickers = {ticker for row in load(VAL) for ticker in row["tickers"]}
    train_tickers = {ticker for row in load(TRAIN) for ticker in row["tickers"]}
    assert not (val_tickers & train_tickers)


def test_indian_market_coverage_in_training():
    rows = [row for row in load(TRAIN) if any(t.endswith(".NS") for t in row["tickers"])]
    assert len(rows) >= 24, len(rows)
    intents = {row["intent"] for row in rows}
    assert {"india_market", "invalid_symbol", "unsupported_india"} <= intents
    currencies = set()
    for row in rows:
        for message in row["messages"]:
            if message["role"] == "tool":
                payload = json.loads(message["content"])["result"]
                if payload.get("data"):
                    currencies.add(payload["data"].get("currency"))
    assert "INR" in currencies


def test_india_heldout_set_is_separate_and_complete():
    rows = load(INDIA)
    assert len(rows) == 32
    assert len({row["question"] for row in rows}) == 32
    intents = {row["intent"] for row in rows}
    for required in (
        "india_quote",
        "india_ratios",
        "india_history",
        "india_news",
        "india_advice",
        "india_prediction",
        "india_insider",
        "india_invalid",
        "india_unsupported",
        "india_stale",
        "india_missing",
    ):
        assert required in intents, required
    frozen = {row["question"] for row in load(TESTSET)}
    assert not ({row["question"] for row in rows} & frozen)
    assert all(row["market"] == "IN" and row["currency"] == "INR" for row in rows)


def test_india_rows_score_with_the_shared_metrics():
    refusal = next(row for row in load(INDIA) if row["intent"] == "india_advice")
    declined = score_prediction(
        refusal, Prediction('{"tool":"finish","args":{}}', "finish", {}, True)
    )
    assert declined["abstention_correct"] is True

    quote = next(row for row in load(INDIA) if row["intent"] == "india_quote")
    text = json.dumps({"tool": "get_stock_quote", "args": {"ticker": quote["tickers"][0]}})
    scored = score_prediction(
        quote,
        Prediction(text, "get_stock_quote", {"ticker": quote["tickers"][0]}, True),
    )
    assert scored["tool_correct"] is True
    assert scored["args_correct"] is True
    assert scored["abstention_correct"] is True
