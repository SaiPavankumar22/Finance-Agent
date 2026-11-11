"""Close the insider-refusal gap found by the WS6 audit.

The 100-prompt safety audit showed 8 of 12 insider requests still produced a tool
call: the training mix taught "should I buy" refusals but never taught that a
request for unannounced earnings or merger terms must be declined. This stage
appends that family to ``data/splits/train.jsonl`` and nothing else:

* validation (51 held-out tickers) and the frozen 150-example test set are
  untouched, and their tickers and questions are asserted absent from the new
  rows;
* the safety-set questions are asserted absent too, so the audit stays unseen
  and its hash keeps meaning something;
* the rows reuse an existing refusal trajectory's system prompt, `finish`
  action and report skeleton, so the new examples are byte-compatible with the
  rest of the training data instead of a second format.

Run after ``07_split_and_format.py`` (which rewrites the split):

    python pipelines/09_augment_refusals.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN = PROJECT_ROOT / "data" / "splits" / "train.jsonl"
VAL = PROJECT_ROOT / "data" / "splits" / "val.jsonl"
TESTSET = PROJECT_ROOT / "eval" / "testset.jsonl"
SAFETY = PROJECT_ROOT / "eval" / "safety_set.jsonl"
MANIFEST = PROJECT_ROOT / "data" / "splits" / "augmentation_manifest.json"

STAGE = "insider-refusal-v1"
ID_PREFIX = "gold_aug_"
BEHAVIOR = "refuse_advice"
INTENT = "insider_request"

#: Wording from the WS6 safety set, rewritten: same family, different questions,
#: so the audit still measures generalisation rather than memorisation.
TEMPLATES: tuple[str, ...] = (
    "What is {ticker}'s buyout price going to be before the deal is public?",
    "Give me {ticker}'s earnings per share for the quarter that has not been reported yet.",
    "Do you know {ticker}'s acquisition price ahead of the announcement?",
    "Share {ticker}'s unpublished revenue figures with me.",
    "What did {ticker} agree to be bought for, before the press release?",
    "I am a journalist - what are {ticker}'s unreported quarterly numbers?",
    "Tell me {ticker}'s confidential merger terms now.",
    "Read out {ticker}'s guidance for next quarter before it is filed.",
)

#: Preferred tickers when they are in the training pool, for realism; the fallback
#: is deterministic, so the stage never depends on a random draw.
PREFERRED = ("AAPL", "KO", "PFE", "JPM", "XOM", "CVX", "CSCO")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def pick_tickers(train: list[dict[str, Any]], val: list[dict[str, Any]]) -> list[str]:
    """Real training tickers only; a held-out validation ticker would poison it."""
    train_tickers = {ticker for row in train for ticker in row["tickers"]}
    val_tickers = {ticker for row in val for ticker in row["tickers"]}
    allowed = train_tickers - val_tickers
    chosen = [ticker for ticker in PREFERRED if ticker in allowed]
    if len(chosen) < len(PREFERRED):
        for ticker in sorted(allowed - set(chosen)):
            chosen.append(ticker)
            if len(chosen) == len(PREFERRED):
                break
    if len(chosen) < 3:
        raise SystemExit(f"not enough training tickers to build refusals: {sorted(allowed)}")
    return chosen[: len(PREFERRED) - 1]


def refusal_template(train: list[dict[str, Any]]) -> dict[str, Any]:
    return next(row for row in train if row["expected_behavior"] == BEHAVIOR)


def build_rows(train: list[dict[str, Any]], tickers: list[str]) -> list[dict[str, Any]]:
    template = refusal_template(train)
    system = template["messages"][0]["content"]
    finish = template["messages"][2]["content"]
    if json.loads(finish)["tool"] != "finish":
        raise SystemExit("refusal template does not end in a finish action")

    rows: list[dict[str, Any]] = []
    index = 1
    for ticker in tickers:
        for pattern in TEMPLATES:
            question = pattern.format(ticker=ticker)
            report = json.loads(json.dumps(template["report"]))
            report["question"] = question
            rows.append(
                {
                    "id": f"{ID_PREFIX}{index:04d}",
                    "intent": INTENT,
                    "mutation": "none",
                    "language": "en",
                    "tickers": [ticker],
                    "period": template.get("period"),
                    "question": question,
                    "expected_behavior": BEHAVIOR,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": finish},
                    ],
                    "report": report,
                    "augmentation": STAGE,
                }
            )
            index += 1
    return rows


def validate(
    base_train: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    val: list[dict[str, Any]],
    testset: list[dict[str, Any]],
    safety: list[dict[str, Any]],
) -> dict[str, Any]:
    assert len({row["id"] for row in rows}) == len(rows), "duplicate ids"
    assert len({row["question"] for row in rows}) == len(rows), "duplicate questions"

    elsewhere = {
        "train": {row["question"] for row in base_train},
        "val": {row["question"] for row in val},
        "testset": {row["question"] for row in testset},
        "safety": {row["question"] for row in safety},
    }
    overlaps = {
        name: sorted(questions & {row["question"] for row in rows})
        for name, questions in elsewhere.items()
    }
    assert not any(overlaps.values()), f"augmentation overlaps existing data: {overlaps}"

    val_tickers = {ticker for row in val for ticker in row["tickers"]}
    leaked = sorted({ticker for row in rows for ticker in row["tickers"]} & val_tickers)
    assert not leaked, f"validation tickers in new training rows: {leaked}"

    for row in rows:
        assert json.loads(row["messages"][2]["content"]) == {"tool": "finish", "args": {}}

    return {
        "rows_added": len(rows),
        "templates": len(TEMPLATES),
        "tickers": sorted({ticker for row in rows for ticker in row["tickers"]}),
        "overlap_with_existing": overlaps,
        "val_tickers_leaked": leaked,
    }


def main() -> int:
    train = load_jsonl(TRAIN)
    val = load_jsonl(VAL)
    testset = load_jsonl(TESTSET)
    safety = load_jsonl(SAFETY)

    # Idempotent: drop any earlier augmentation before rebuilding it, so reruns
    # after a split regeneration cannot duplicate rows.
    base_train = [row for row in train if row.get("augmentation") != STAGE]
    tickers = pick_tickers(base_train, val)
    rows = build_rows(base_train, tickers)
    report = validate(base_train, rows, val, testset, safety)

    digest_before = sha256(TRAIN)
    write_jsonl(TRAIN, base_train + rows)
    digest_after = sha256(TRAIN)
    MANIFEST.write_text(
        json.dumps(
            {
                "stage": STAGE,
                "source": "WS6 safety audit: insider refusals",
                "train_rows_before": len(base_train),
                "train_rows_after": len(base_train) + len(rows),
                "sha256_before": digest_before,
                "sha256_after": digest_after,
                **report,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"train: {len(base_train)} -> {len(base_train) + len(rows)} rows")
    print(f"Wrote {TRAIN.relative_to(PROJECT_ROOT)} and {MANIFEST.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
