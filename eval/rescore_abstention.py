"""Rescore the cached WS3 baselines with a real abstention metric.

The first baseline reported `abstention_accuracy`, which is overall accuracy
over a set with 37 abstain rows and 113 healthy ones: a model that never
abstains scores 0.7533 and looks acceptable. This script re-derives every
prediction from the cached raw text (no API calls, no cost), replaces the
summary with precision/recall on the abstain class, and writes
``eval/results/abstention.json``.

    python -m eval.rescore_abstention
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.metrics import Prediction, score_prediction, summarize
from eval.run_baseline import SYSTEM_PROMPT, parse_prediction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTSET = PROJECT_ROOT / "eval" / "testset.jsonl"
BASELINE = PROJECT_ROOT / "eval" / "results" / "baseline.json"
OUTPUT = PROJECT_ROOT / "eval" / "results" / "abstention.json"


def load_rows() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in TESTSET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rescore_model(model: dict[str, Any], rows_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Rebuild scored rows from cached text so the decision rule is current."""
    scored: list[dict[str, Any]] = []
    for prediction in model.get("predictions") or []:
        row = rows_by_id.get(prediction["id"])
        if row is None:
            continue
        rebuilt: Prediction = parse_prediction(prediction["raw_text"])
        scored.append(score_prediction(row, rebuilt))
    return {
        "model": model["model"],
        "status": model.get("status", "ok"),
        "summary": summarize(scored),
        "predictions": scored,
    }


def ceiling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How much abstain recall a turn-one model can reach by following gold.

    For 23 of the 37 abstain rows the gold first action is a tool call; the
    decision to abstain is only justified after that tool returns an error or
    missing data. A turn-one judge therefore cannot credit those rows unless it
    also accepts an early refusal, so this ceiling is reported alongside recall
    instead of being silently baked into the denominator.
    """
    abstain_rows = [row for row in rows if row.get("expect_abstain")]
    finish_first = [row for row in abstain_rows if row["gold_tools"][0] == "finish"]
    return {
        "abstain_rows": len(abstain_rows),
        "gold_declines_at_turn_one": len(finish_first),
        "gold_calls_tool_first": len(abstain_rows) - len(finish_first),
        "max_recall_if_following_gold": round(len(finish_first) / len(abstain_rows), 4)
        if abstain_rows
        else None,
        "intents_declining_at_turn_one": sorted({row["intent"] for row in finish_first}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    rows = load_rows()
    rows_by_id = {row["id"]: row for row in rows}
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    rescored: list[dict[str, Any]] = []
    for model in baseline["models"]:
        if not model.get("predictions"):
            rescored.append(
                {
                    "model": model["model"],
                    "status": model.get("status", "unavailable"),
                    "summary": {"examples": 0},
                    "abstention": {"scored": False, "reason": "no predictions"},
                    "error": model.get("error"),
                }
            )
            continue
        result = rescore_model(model, rows_by_id)
        rescored.append(
            {
                "model": result["model"],
                "status": result["status"],
                "summary": result["summary"],
                "abstention": result["summary"]["abstention"],
            }
        )

    prompt_offers_refusal = any(
        marker in SYSTEM_PROMPT.lower() for marker in ("refus", "abstain", "cannot answer")
    )
    report = {
        "schema_version": "1.1",
        "testset": str(TESTSET.relative_to(PROJECT_ROOT)),
        "examples": len(rows),
        "metric": "abstention is scored as a binary decision: recall/precision on the "
        "37 abstain rows, with over-abstention measured on the 113 others",
        "turn_one_ceiling": ceiling(rows),
        "validity": {
            "planner_prompt_offers_refusal": prompt_offers_refusal,
            "note": "The baseline prompts 'Never answer the user directly. Emit exactly one "
            "JSON object', so a refusal is off-contract for this harness. A recall of 0.0 "
            "measures the scoring setup as much as the model: turn-one scoring cannot "
            "measure abstention for the 23 rows that need a tool result, nor offer a legal "
            "refusal for the 14 advice rows.",
            "consequence": "WS6 must score abstention after tool errors on a replayed "
            "trajectory, with an explicit finish/refusal affordance in the prompt.",
        },
        "models": rescored,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Keep baseline.json consistent: same predictions, corrected summaries.
    for model in baseline["models"]:
        match = next((item for item in rescored if item["model"] == model["model"]), None)
        if match and model.get("predictions"):
            model["summary"] = match["summary"]
    baseline["schema_version"] = "1.1"
    baseline["rescored_with"] = "eval.rescore_abstention"
    baseline["metric_note"] = (
        "abstention_accuracy (overall accuracy) was removed; it scored 0.7533 for "
        "models that never abstained. Use summary.abstention.recall/precision."
    )
    BASELINE.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(
            f"{'model':32} {'recall':>7} {'95% CI':>14} {'prec':>5} "
            f"{'spec':>5} {'bal':>5} {'over':>5} {'under':>5}"
        )
        for item in rescored:
            block = item.get("abstention") or {}
            if not block.get("scored"):
                print(f"{item['model']:32} {'n/a':>7}  ({item.get('status')})")
                continue
            ci = block["recall_95ci"] or [0, 0]
            precision = "n/a" if block["precision"] is None else f"{block['precision']:.3f}"
            print(
                f"{item['model']:32} {block['recall']:>7.3f} "
                f"{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>14} {precision:>5} "
                f"{block['specificity']:>5.3f} {block['balanced_accuracy']:>5.3f} "
                f"{block['over_abstention_rate']:>5.3f} {block['under_abstention_rate']:>5.3f}"
            )
        floor = report["turn_one_ceiling"]
        print(
            f"\nturn-one ceiling: {floor['gold_declines_at_turn_one']}/{floor['abstain_rows']} "
            f"abstain rows can be declined before any tool call "
            f"(max recall {floor['max_recall_if_following_gold']})"
        )
        for item in rescored:
            block = item.get("abstention") or {}
            if block.get("scored"):
                print(f"\n{item['model']} refusal rate by abstain intent:")
                for intent, counts in block["recall_by_abstain_intent"].items():
                    print(f"  {intent:20} {counts['declined']}/{counts['total']}")
    print(f"Saved {OUTPUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
