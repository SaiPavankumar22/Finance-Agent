"""Compare a Hub re-score slice with the accepted artifacts, per row.

    python scripts/check_reproduce.py artifacts/sft/reproduce-5ae2ee8.json

Exits non-zero when the published weights disagree with the accepted numbers on
any row of the slice, so a mismatched upload cannot pass as a reproduction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    reproduce_path = Path(argv[1] if len(argv) > 1 else "artifacts/sft/reproduce-5ae2ee8.json")
    repro = load(reproduce_path)

    accepted_eval = load(Path("artifacts/sft/eval-ic7.json"))
    accepted_safety = load(Path("artifacts/sft/safety-ic7.json"))

    frozen_accepted = {row["id"]: row for row in accepted_eval["predictions"]["test"]["tuned"]}
    safety_accepted = {row["id"]: row for row in accepted_safety["predictions"]["tuned"]}

    failures: list[str] = []
    for row in repro["frozen"]["predictions"]:
        reference = frozen_accepted[row["id"]]
        for key in ("predicted_tool", "tool_correct", "args_correct", "schema_valid"):
            if row.get(key) != reference.get(key):
                failures.append(f"frozen {row['id']} {key}: {row.get(key)} != {reference.get(key)}")

    for row in repro["safety"]["predictions"]:
        reference = safety_accepted[row["id"]]
        for key in ("decision", "declined", "raw_text"):
            if row.get(key) != reference.get(key):
                failures.append(f"safety {row['id']} {key} differs from the accepted run")

    frozen = repro["frozen"]
    safety = repro["safety"]
    print(
        json.dumps(
            {
                "adapter": repro["adapter"],
                "frozen_slice": {
                    "rows": frozen["rows"],
                    "tool_selection_accuracy": frozen["tool_selection_accuracy"],
                    "argument_accuracy": frozen["argument_accuracy"],
                    "schema_valid_rate": frozen["schema_valid_rate"],
                },
                "safety_slice": {
                    "rows": safety["rows"],
                    "refusal_rows": safety["refusal_rows"],
                    "declined": safety["declined"],
                    "refusal_recall": safety["refusal_recall"],
                    "over_refusal": safety["over_refusal"],
                    "unsafe_text_rows": safety["unsafe_text_rows"],
                },
                "per_row_mismatches": len(failures),
            },
            indent=2,
        )
    )
    for failure in failures[:20]:
        print("MISMATCH", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
