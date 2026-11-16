"""Re-score the accepted ic7 run with AuditKIT instead of our own metrics.

This is the audit half of the LexiLabs chain: `pipelines/06_curate.py` produced
the data with CuratorKIT, `training/train_full.py` trained the adapter, and this
script hands the recorded per-row predictions to `auditkit.evaluate()` so an
independent spine reproduces the numbers and attaches its own run fingerprint.

Two AuditKIT metrics carry real work here:

* ``exact_match`` on the chosen tool name - the same quantity the gate report
  calls tool-selection accuracy.
* ``is_json(require_keys=["tool", "args"])`` - a code check over the model's raw
  text, the same quantity the gate report calls schema validity.

AuditKIT's ``compare()`` then runs its own paired significance test on base vs
tuned, which nothing in this repo computed before.

Run:
    python -m scripts.audit_with_auditkit
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import auditkit as ak

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "artifacts/sft/eval-ic7.json"
DEFAULT_TESTSET = ROOT / "eval/testset.jsonl"
DEFAULT_OUT = ROOT / "artifacts/audit/auditkit-ic7.json"


def load_testset(path: Path) -> dict[str, dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return {row["id"]: row for row in rows}


def samples_for(
    predictions: list[dict[str, Any]], testset: dict[str, dict[str, Any]]
) -> tuple[list[ak.Sample], list[ak.Sample]]:
    """Tool-name samples (for exact_match) and raw-text samples (for is_json)."""

    tools: list[ak.Sample] = []
    texts: list[ak.Sample] = []
    for row in predictions:
        gold = testset.get(row["id"])
        if gold is None:  # predictions must stay inside the frozen test set
            raise SystemExit(f"prediction {row['id']} is not in the frozen test set")
        metadata = {
            "intent": gold["intent"],
            "expected_behavior": gold["expected_behavior"],
            "schema_valid": row["schema_valid"],
            "args_correct": row["args_correct"],
            "tool_correct": row["tool_correct"],
        }
        tools.append(
            ak.Sample(
                id=row["id"],
                input=gold["question"],
                target=row["expected_tool"],
                actual_output=row["predicted_tool"],
                metadata=metadata,
            )
        )
        texts.append(
            ak.Sample(
                id=row["id"],
                input=gold["question"],
                target=row["expected_tool"],
                actual_output=row["raw_text"],
                metadata=metadata,
            )
        )
    return tools, texts


def audit(
    predictions_path: Path, testset_path: Path, out_path: Path, split: str = "test"
) -> dict[str, Any]:
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    testset = load_testset(testset_path)
    arms = payload["predictions"][split]

    results: dict[str, ak.RunResult] = {}
    for arm in ("base", "tuned"):
        tools, texts = samples_for(arms[arm], testset)
        by_tool = ak.evaluate(tools, "precomputed", ["exact_match"])
        by_text = ak.evaluate(texts, "precomputed", [ak.IsJson(require_keys=["tool", "args"])])
        results[f"{arm}:tool_selection"] = by_tool
        results[f"{arm}:schema"] = by_text

    comparison = ak.compare(
        results["base:tool_selection"], results["tuned:tool_selection"]
    )

    record = {
        "auditkit_version": getattr(ak, "__version__", "unknown"),
        "source_predictions": str(predictions_path.relative_to(ROOT)),
        "split": split,
        "rows": len(arms["tuned"]),
        "runs": {
            name: {
                "headline": {k: round(float(v), 4) for k, v in run.headline.items()},
                "fingerprint": run.fingerprint,
                "failed": run.failed_count,
            }
            for name, run in results.items()
        },
        "paired_comparison": {
            "metric": "exact_match (tool selection)",
            "summary": comparison.summary(),
            "significance": comparison.significance(),
            "metric_deltas": json.loads(
                json.dumps([asdict(delta) for delta in comparison.metric_deltas()], default=str)
            ),
            "improved": len(comparison.improved()),
            "regressed": len(comparison.regressed()),
        },
        "cross_check": {
            arm: {
                "tool_selection": {
                    "auditkit": round(
                        float(results[f"{arm}:tool_selection"].headline["exact_match"]), 4
                    ),
                    "repo_gate_report": payload["eval"][split][arm]["tool_selection_accuracy"],
                    "agree": abs(
                        float(results[f"{arm}:tool_selection"].headline["exact_match"])
                        - payload["eval"][split][arm]["tool_selection_accuracy"]
                    )
                    < 1e-4,
                },
                "schema_valid": {
                    "auditkit": round(
                        float(results[f"{arm}:schema"].headline["is_json(keys=tool,args)"]), 4
                    ),
                    "repo_gate_report": payload["eval"][split][arm]["schema_valid_rate"],
                    "agree": abs(
                        float(results[f"{arm}:schema"].headline["is_json(keys=tool,args)"])
                        - payload["eval"][split][arm]["schema_valid_rate"]
                    )
                    < 1e-4,
                },
            }
            for arm in ("base", "tuned")
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", default="test", choices=("test", "val"))
    args = parser.parse_args()

    record = audit(args.predictions, args.testset, args.out, args.split)
    print(f"rows: {record['rows']}  AuditKIT {record['auditkit_version']}")
    for name, run in record["runs"].items():
        print(f"  {name:24s} {run['headline']}  sha {run['fingerprint'][:16]}")
    for arm, checks in record["cross_check"].items():
        for name, check in checks.items():
            flag = "agrees" if check["agree"] else "DIFFERS"
            print(
                f"  {arm:6s} {name:14s} AuditKIT {check['auditkit']:.4f} vs report "
                f"{check['repo_gate_report']:.4f} - {flag}"
            )
    paired = record["paired_comparison"]
    print("paired comparison (exact_match, base -> tuned):")
    print(f"  {paired['summary']}")
    print(
        f"  improved {paired['improved']} / regressed {paired['regressed']} rows; "
        f"significance {paired['significance']}"
    )
    print(f"written: {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
