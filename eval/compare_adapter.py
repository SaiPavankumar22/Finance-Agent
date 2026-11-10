"""Compare the tuned adapter against the Tiny Aya Global baseline.

Reads `artifacts/sft/full.json` (written by `training/train_full.py`) and the
historical WS3 baselines, then prints and saves a before/after table.

Two comparisons are reported because they answer different questions:

- **Local base vs tuned** use the identical prompt, parser and decoding, so the
  difference is the adapter alone.
- **Historical provider baseline** (WS3, `eval/results/baseline.json`) is the
  number the plan was written against; it ran fp16 on a hosted provider, so it
  is context, not a controlled row.

    python -m eval.compare_adapter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
#: Result of `modal run training/train_full.py::eval_main`: one prompt, one
#: decode rule, adapter on and off. The training run's own payload is kept for
#: the training statistics but scored under a smaller token budget.
DEFAULT_RESULT = PROJECT_ROOT / "artifacts" / "sft" / "eval-json-stop.json"
BASELINE = PROJECT_ROOT / "eval" / "results" / "baseline.json"
OUTPUT = PROJECT_ROOT / "artifacts" / "sft" / "comparison.md"
PROVIDER_MODEL = "CohereLabs/tiny-aya-global"


def fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def abstention_bits(summary: dict[str, Any]) -> str:
    block = summary.get("abstention") or {}
    if not block.get("scored"):
        return "n/a"
    return (
        f"R {fmt(block['recall'], 2)} / P {fmt(block['precision'], 2)} / "
        f"spec {fmt(block['specificity'], 2)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result_path = Path(args.result)
    full = json.loads(result_path.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    provider = next(
        (model for model in baseline["models"] if model["model"] == PROVIDER_MODEL), None
    )
    provider_summary = (provider or {}).get("summary") or {}

    lines: list[str] = [
        "# Adapter vs baseline",
        "",
        f"Adapter: `{full['adapter']}`.",
        f"Decoding: {full.get('decode', 'greedy')} "
        f"(max {full.get('max_new_tokens')} new tokens).",
        "",
        "| Set | System | Tool selection | Argument acc. | Schema valid | Abstention |",
        "|---|---|---|---|---|---|",
    ]

    for split in ("val", "test"):
        for name in ("base", "tuned"):
            summary = full["eval"][split][name]
            lines.append(
                f"| {split} | {name} (same prompt) | {fmt(summary['tool_selection_accuracy'])} "
                f"| {fmt(summary['argument_accuracy'])} | {fmt(summary['schema_valid_rate'])} "
                f"| {abstention_bits(summary)} |"
            )

    if provider_summary.get("examples"):
        lines.append(
            f"| test | provider baseline (fp16, WS3) | "
            f"{fmt(provider_summary['tool_selection_accuracy'])} "
            f"| {fmt(provider_summary['argument_accuracy'])} "
            f"| {fmt(provider_summary['schema_valid_rate'])} "
            f"| {abstention_bits(provider_summary)} |"
        )

    lines.extend(["", "## Paired per-example changes (test)", ""])
    paired = full["eval"]["paired"]["test"]
    for metric, counts in paired.items():
        lines.append(
            f"- {metric}: gained {counts['gained']}, lost {counts['lost']}, "
            f"correct in both {counts['both']}, wrong in both {counts['neither']}"
        )

    lines.extend(["", "## Per-intent tool selection", ""])
    lines.append("| Intent | val base | val tuned | test base | test tuned |")
    lines.append("|---|---|---|---|---|")
    intents = sorted(
        {
            intent
            for split in ("val", "test")
            for name in ("base", "tuned")
            for intent in full["eval"][split][name]["by_intent"]
        }
    )

    def intent_rate(split: str, name: str, intent: str) -> str:
        entry = full["eval"][split][name]["by_intent"].get(intent)
        return "—" if entry is None else fmt(entry["tool_selection_accuracy"], 2)

    for intent in intents:
        lines.append(
            f"| {intent} | {intent_rate('val', 'base', intent)} "
            f"| {intent_rate('val', 'tuned', intent)} "
            f"| {intent_rate('test', 'base', intent)} "
            f"| {intent_rate('test', 'tuned', intent)} |"
        )

    checks = full.get("data_checks", {})
    train = full.get("train", {})
    lines.extend(["", "## Data integrity", ""])
    if checks:
        lines.append(
            f"- train / val / test rows: {checks.get('train_rows')} / "
            f"{checks.get('val_rows')} / {checks.get('test_rows')}"
        )
        lines.append(
            f"- val questions in train: {checks.get('val_questions_in_train')}; "
            f"val tickers in train: {len(checks.get('val_tickers_in_train') or [])}"
        )
        lines.append(f"- test questions in train: {checks.get('test_questions_in_train')}")
    else:
        lines.append(
            f"- val / test rows evaluated: {full['val_rows']} / {full['test_rows']}"
        )
    lines.append(
        f"- frozen test set unchanged: "
        f"{full['testset_sha256']['before'] == full['testset_sha256']['after']}"
    )
    if train:
        lines.append(f"- rows truncated by max_len: {train['rows_over_max_len']}")
        lines.append(
            f"- best epoch {train['best_epoch']} of {full['config']['epochs']}: "
            f"{train['epoch_val']}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.quiet:
        print("\n".join(lines))
    print(f"\nSaved {OUTPUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
