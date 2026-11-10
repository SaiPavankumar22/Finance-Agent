"""Metrics for the first-turn WS3 tool-selection baseline."""

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class Prediction:
    raw_text: str
    tool: str | None
    args: dict[str, Any]
    schema_valid: bool


def expected_first_call(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool = row["gold_tools"][0]
    ticker = row["tickers"][0]
    if tool == "get_historical_prices":
        return tool, {"ticker": ticker, "period": row.get("period") or "1y"}
    if tool == "get_company_news":
        return tool, {"ticker": ticker, "limit": 5}
    if tool in {"get_stock_quote", "get_financial_ratios"}:
        return tool, {"ticker": ticker}
    return tool, {}


def score_prediction(row: dict[str, Any], prediction: Prediction) -> dict[str, Any]:
    expected_tool, expected_args = expected_first_call(row)
    args_correct = prediction.args == expected_args
    abstention_expected = bool(row.get("expect_abstain"))
    abstention_signal = (
        prediction.tool == "finish"
        or any(word in prediction.raw_text.lower() for word in ("abstain", "refuse", "unavailable"))
    )
    return {
        "id": row["id"],
        "intent": row["intent"],
        "expected_tool": expected_tool,
        "predicted_tool": prediction.tool,
        "schema_valid": prediction.schema_valid,
        "tool_correct": prediction.tool == expected_tool,
        "args_correct": args_correct,
        "abstention_expected": abstention_expected,
        "abstention_signal": abstention_signal,
        "abstention_correct": abstention_signal == abstention_expected,
        "raw_text": prediction.raw_text,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {"examples": 0}

    def rate(key: str) -> float:
        return round(sum(bool(row[key]) for row in rows) / count, 4)

    return {
        "examples": count,
        "schema_valid_rate": rate("schema_valid"),
        "tool_selection_accuracy": rate("tool_correct"),
        "argument_accuracy": rate("args_correct"),
        "abstention_accuracy": rate("abstention_correct"),
        "by_intent": {
            intent: {
                "examples": len(group),
                "tool_selection_accuracy": round(
                    sum(bool(row["tool_correct"]) for row in group) / len(group), 4
                ),
                "schema_valid_rate": round(
                    sum(bool(row["schema_valid"]) for row in group) / len(group), 4
                ),
            }
            for intent in sorted({row["intent"] for row in rows})
            for group in [[item for item in rows if item["intent"] == intent]]
        },
        "predicted_tools": dict(Counter(row["predicted_tool"] for row in rows)),
    }
