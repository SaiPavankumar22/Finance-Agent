"""Turn a WS6 safety run into the safety delta table (plan section 13).

The numbers come from a GPU run of ``modal run training/train_full.py::safety_main``,
which scores the base model and the adapter on ``eval/safety_set.jsonl`` and
writes ``artifacts/sft/safety.json``. This script only reads that file, so the
table can be regenerated without a GPU:

    python -m eval.run_safety
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.safety import load_rows, safety_drift, safety_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT = PROJECT_ROOT / "artifacts" / "sft" / "safety.json"
OUTPUT = PROJECT_ROOT / "artifacts" / "safety" / "comparison.md"


def number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def change(base: float | None, tuned: float | None, digits: int = 3) -> str:
    if base is None or tuned is None:
        return "n/a"
    return f"{tuned - base:+.{digits}f}"


def interval(values: list[float] | None) -> str:
    return "n/a" if not values else f"[{values[0]:.2f}, {values[1]:.2f}]"


def table(base: dict[str, Any], tuned: dict[str, Any]) -> list[str]:
    lines = [
        "| Metric | Base | Tuned | Delta |",
        "|---|---|---|---|",
    ]
    rows = (
        (
            "Refusal recall (74 rows)",
            base["refusal"]["recall"],
            tuned["refusal"]["recall"],
        ),
        (
            "Over-refusal rate (26 controls)",
            base["benign"]["over_refusal_rate"],
            tuned["benign"]["over_refusal_rate"],
        ),
        (
            "Post-error declination (10 rows)",
            base["error_handling"]["handled_rate"],
            tuned["error_handling"]["handled_rate"],
        ),
        (
            "Benign tool selection (26 rows)",
            base["benign"]["tool_accuracy"],
            tuned["benign"]["tool_accuracy"],
        ),
    )
    for label, before, after in rows:
        lines.append(
            f"| {label} | {number(before)} | {number(after)} | {change(before, after)} |"
        )
    ci = tuned["refusal"]["recall_95ci"]
    lines.append("")
    lines.append(f"Tuned refusal recall 95% CI: {interval(ci)}.")
    lines.append(
        f"Unsafe text rows: base {base['unsafe_text_rows']}, "
        f"tuned {tuned['unsafe_text_rows']}. "
        f"Repeated failing calls after two errors: "
        f"base {base['error_handling']['repeated_failing_call']}, "
        f"tuned {tuned['error_handling']['repeated_failing_call']}."
    )
    return lines


def category_table(base: dict[str, Any], tuned: dict[str, Any]) -> list[str]:
    lines = ["", "| Category | Base recall | Tuned recall | Delta |", "|---|---|---|---|"]
    for category, counts in tuned["refusal"]["by_category"].items():
        before = base["refusal"]["by_category"].get(category, {}).get("recall")
        lines.append(
            f"| {category} | {number(before)} | {number(counts['recall'])} "
            f"| {change(before, counts['recall'])} |"
        )
    return lines


def examples(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]], limit: int = 5
) -> list[str]:
    by_id = {row["id"]: row for row in rows}
    lines = ["", "### Tuned-model failures", ""]
    violations = [
        item
        for item in predictions
        if item["expect_refusal"] and not item["declined"]
    ][:limit]
    if not violations:
        lines.append("No compliance violations: every request for advice, a prediction "
                     "or non-public information was declined.")
    for item in violations:
        row = by_id[item["id"]]
        excerpt = " ".join(item["raw_text"].split())[:160]
        lines.append(f"- `{item['id']}` ({item['category']}): {row['question']}")
        lines.append(f"  - model said: `{excerpt}`")
    repeats = [
        item
        for item in predictions
        if item.get("after_error", {}).get("repeated_failing_call")
    ][:limit]
    if repeats:
        lines.append("")
        lines.append("Repeated the call that had already failed twice:")
        for item in repeats:
            lines.append(f"- `{item['id']}`: {by_id[item['id']]['question']}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default=str(RESULT))
    args = parser.parse_args()

    payload = json.loads(Path(args.result).read_text(encoding="utf-8"))
    rows = load_rows()
    digest = payload.get("safety_set", {}).get("sha256", "n/a")[:12]

    # Recompute from the raw predictions so the table always reflects the
    # current scoring rule, the way eval.rescore_abstention does for WS3.
    scored = payload.get("predictions") or {}
    if scored.get("base") and scored.get("tuned"):
        base = safety_report(rows, scored["base"])
        tuned = safety_report(rows, scored["tuned"])
        drift = safety_drift(base, tuned)
    else:
        base = payload["eval"]["base"]
        tuned = payload["eval"]["tuned"]
        drift = payload["drift"]

    lines = [
        "# WS6 safety evaluation",
        "",
        f"Set: `eval/safety_set.jsonl` ({len(rows)} prompts, {digest}), "
        "replayed on recorded tool results, same prompt and decode rule for both arms.",
        "",
        f"Decode: {payload.get('decode', 'greedy, stop at first complete JSON object')}.",
        "",
        f"**Verdict: {drift['verdict']}.**",
        "",
    ]
    if drift["reasons"]:
        lines.append("Triggers:")
        lines.extend(f"- {reason}" for reason in drift["reasons"])
        lines.append("")
    else:
        lines.append(
            "Every drift check is inside its threshold: refusal recall did not fall by "
            f"{abs(drift['thresholds']['refusal_recall_drop']):.2f}, over-refusal did not "
            f"rise by {drift['thresholds']['over_refusal_rise']:.2f}, post-error "
            "declination held, and no new unsafe text appeared."
        )
        lines.append("")
    lines.extend(table(base, tuned))
    lines.extend(category_table(base, tuned))
    lines.extend(examples(rows, payload.get("predictions", {}).get("tuned", [])))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(table(base, tuned)))
    print(f"\n{drift['verdict']}")
    for reason in drift["reasons"]:
        print(f"  - {reason}")
    print(f"Saved {OUTPUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
