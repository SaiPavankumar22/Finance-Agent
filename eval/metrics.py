"""Metrics for tool-planning baselines and the abstention classifier.

Abstention is scored as a two-class decision, not as overall accuracy: with 37
abstain rows in a 150-row set, "never abstain" already scores 0.7533 overall
accuracy, which hides a total recall failure. The report therefore returns
precision, recall, specificity and balanced accuracy, plus over- and
under-abstention rates.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

DECISION_ABSTAIN = "abstain"
DECISION_CALL = "call_tool"
DECISION_MALFORMED = "malformed"

REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "cannot provide",
    "unable to",
    "not able to",
    "no data",
    "not available",
    "insufficient",
    "abstain",
    "refuse",
)
REFUSAL_SCAN_CHARS = 400
Z_95 = 1.959963984540054


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


def abstention_decision(prediction: Prediction) -> str:
    """Classify a turn-one output as declining, gathering evidence, or broken.

    A structured ``finish`` call is the only unambiguous abstention: the model
    chose to stop without evidence. Unparseable text counts as abstention only
    when it opens with a refusal phrase, because a malformed tool call is a
    schema failure, not a decision to decline, and must not be scored as one.
    """
    if prediction.schema_valid:
        return DECISION_ABSTAIN if prediction.tool == "finish" else DECISION_CALL
    head = prediction.raw_text[:REFUSAL_SCAN_CHARS].lower()
    if any(marker in head for marker in REFUSAL_MARKERS):
        return DECISION_ABSTAIN
    return DECISION_MALFORMED


def score_prediction(row: dict[str, Any], prediction: Prediction) -> dict[str, Any]:
    expected_tool, expected_args = expected_first_call(row)
    decision = abstention_decision(prediction)
    abstention_expected = bool(row.get("expect_abstain"))
    declined = decision == DECISION_ABSTAIN
    return {
        "id": row["id"],
        "intent": row["intent"],
        "expected_tool": expected_tool,
        "predicted_tool": prediction.tool,
        "schema_valid": prediction.schema_valid,
        "tool_correct": prediction.tool == expected_tool,
        "args_correct": prediction.args == expected_args,
        "abstention_expected": abstention_expected,
        "abstention_decision": decision,
        "abstention_signal": declined,
        "abstention_correct": declined == abstention_expected,
        "raw_text": prediction.raw_text,
    }


def wilson_interval(successes: int, total: int, z: float = Z_95) -> list[float] | None:
    """Wilson score interval, used because bootstrap collapses at 0/37."""
    if total <= 0:
        return None
    fraction = successes / total
    denominator = 1 + z * z / total
    centre = fraction + z * z / (2 * total)
    margin = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total * total))
    low = max(0.0, (centre - margin) / denominator)
    high = min(1.0, (centre + margin) / denominator)
    return [round(low, 4), round(high, 4)]


def abstention_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Precision/recall on the abstain class, with over-abstention measured."""
    decisions = [row.get("abstention_decision") for row in rows]
    if not rows or any(decision is None for decision in decisions):
        return {"scored": False, "reason": "predictions lack abstention_decision; rescore first"}

    def count(declined: bool, expected: bool) -> int:
        return sum(
            1
            for row in rows
            if (row["abstention_decision"] == DECISION_ABSTAIN) is declined
            and row["abstention_expected"] is expected
        )

    true_positive = count(True, True)
    false_positive = count(True, False)
    false_negative = count(False, True)
    true_negative = count(False, False)
    positives = true_positive + false_negative
    negatives = true_negative + false_positive
    predicted_positive = true_positive + false_positive

    recall = true_positive / positives if positives else None
    precision = true_positive / predicted_positive if predicted_positive else None
    specificity = true_negative / negatives if negatives else None
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    balanced = None if recall is None or specificity is None else (recall + specificity) / 2

    def per_intent(expected: bool) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, int]] = {}
        for row in rows:
            if row["abstention_expected"] is not expected:
                continue
            bucket = grouped.setdefault(row["intent"], {"declined": 0, "total": 0})
            bucket["total"] += 1
            bucket["declined"] += int(row["abstention_decision"] == DECISION_ABSTAIN)
        return {
            intent: {**counts, "rate": round(counts["declined"] / counts["total"], 4)}
            for intent, counts in sorted(grouped.items())
        }

    return {
        "scored": True,
        "abstain_rows": positives,
        "non_abstain_rows": negatives,
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
        "specificity": None if specificity is None else round(specificity, 4),
        "balanced_accuracy": None if balanced is None else round(balanced, 4),
        "over_abstention_rate": round(false_positive / negatives, 4) if negatives else None,
        "under_abstention_rate": round(false_negative / positives, 4) if positives else None,
        "recall_95ci": wilson_interval(true_positive, positives),
        "precision_95ci": wilson_interval(true_positive, predicted_positive),
        "no_positive_predictions": predicted_positive == 0,
        "malformed_rows": sum(
            1 for row in rows if row["abstention_decision"] == DECISION_MALFORMED
        ),
        "recall_by_abstain_intent": per_intent(True),
        "over_refusal_by_intent": per_intent(False),
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
        "abstention": abstention_report(rows),
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
