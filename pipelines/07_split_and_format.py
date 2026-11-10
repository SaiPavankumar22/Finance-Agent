"""Split the curated WS4 set into train/val and write the dataset card.

The frozen evaluation set is never touched here. Tickers are held out as
groups so validation measures behaviour on instruments the model has not seen,
which is the only split that means anything for a tool-calling agent: question
text overlap is already zero by construction.

Usage:
    python pipelines/07_split_and_format.py
    python pipelines/07_split_and_format.py --val-tickers 15 --val-target 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "data" / "curated" / "curated.jsonl"
TESTSET = PROJECT_ROOT / "eval" / "testset.jsonl"
SEED_DIR = PROJECT_ROOT / "data" / "seeds"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
TRAIN = OUTPUT_DIR / "train.jsonl"
VAL = OUTPUT_DIR / "val.jsonl"
MANIFEST = OUTPUT_DIR / "split_manifest.json"
CARD = OUTPUT_DIR / "dataset_card.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def known_symbols() -> set[str]:
    payload = json.loads((SEED_DIR / "tickers.json").read_text(encoding="utf-8"))
    return {item["symbol"] for item in payload["tickers"]}


def group_of(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(record["tickers"]))


def split(
    rows: list[dict[str, Any]],
    val_tickers: int,
    val_target: int,
    bogus_holdout: int,
    max_per_cell: int,
    seed: int,
) -> dict[str, Any]:
    real = known_symbols()
    rng = random.Random(seed)

    by_ticker: dict[str, int] = Counter(t for row in rows for t in row["tickers"])
    real_candidates = sorted(t for t in by_ticker if t in real and by_ticker[t] >= 3)
    bogus_candidates = sorted(t for t in by_ticker if t not in real)
    if len(real_candidates) < val_tickers:
        raise RuntimeError(
            f"only {len(real_candidates)} real tickers have >=3 rows; "
            f"cannot hold out {val_tickers}"
        )

    held_real = sorted(rng.sample(real_candidates, val_tickers))
    held_bogus = sorted(rng.sample(bogus_candidates, min(bogus_holdout, len(bogus_candidates))))
    holdout = set(held_real) | set(held_bogus)

    val_pool: list[dict[str, Any]] = []
    train_pool: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for record in rows:
        tickers = set(record["tickers"])
        if tickers <= holdout:
            val_pool.append(record)
        elif tickers & holdout:
            # Mixed groups would leak a held-out ticker into training.
            dropped.append(record)
        else:
            train_pool.append(record)

    # Round-robin over intents so validation is not dominated by one category.
    val_pool.sort(key=lambda item: item["id"])
    per_intent = defaultdict(list)
    for record in val_pool:
        per_intent[record["intent"]].append(record)
    quota = max(1, val_target // max(1, len(per_intent)))
    val: list[dict[str, Any]] = []
    for intent in sorted(per_intent):
        val.extend(per_intent[intent][:quota])

    # Cap repeated (intent, ticker-group) cells so one template pair cannot
    # dominate the training set.
    cell_counts: Counter[tuple[str, tuple[str, ...]]] = Counter()
    train: list[dict[str, Any]] = []
    capped: list[dict[str, Any]] = []
    for record in sorted(train_pool, key=lambda item: item["id"]):
        cell = (record["intent"], group_of(record))
        if cell_counts[cell] >= max_per_cell:
            capped.append(record)
            continue
        cell_counts[cell] += 1
        train.append(record)

    frozen = {row["question"] for row in load_jsonl(TESTSET)} if TESTSET.exists() else set()
    train_questions = {row["question"] for row in train}
    val_questions = {row["question"] for row in val}

    train_tickers = {t for row in train for t in row["tickers"]}
    val_tickers_seen = {t for row in val for t in row["tickers"]}
    shared_questions = train_questions & val_questions
    leaked = (train_questions | val_questions) & frozen
    ticker_leak = train_tickers & val_tickers_seen
    missing_intents = {row["intent"] for row in rows} - {row["intent"] for row in train}

    if shared_questions:
        raise RuntimeError(f"train/val question overlap: {len(shared_questions)}")
    if leaked:
        raise RuntimeError(f"{len(leaked)} split rows appear in the frozen test set")
    if ticker_leak:
        raise RuntimeError(f"held-out tickers leak into train: {sorted(ticker_leak)}")
    if missing_intents:
        raise RuntimeError(f"intents missing from train: {sorted(missing_intents)}")

    return {
        "train": train,
        "val": val,
        "held_real": held_real,
        "held_bogus": held_bogus,
        "dropped": dropped,
        "capped": capped,
        "frozen": frozen,
        "train_tickers": train_tickers,
        "val_tickers": val_tickers_seen,
    }


def write_card(result: dict[str, Any], args: argparse.Namespace) -> None:
    train, val = result["train"], result["val"]
    rows = train + val
    intents = sorted({row["intent"] for row in rows})
    behaviors = sorted({row["expected_behavior"] for row in rows})
    frozen_overlap = len(
        result["train_tickers"] & {t for row in load_jsonl(TESTSET) for t in row["tickers"]}
    )
    per_intent = Counter(row["intent"] for row in train)
    lines = [
        "# Dataset Card: finance agent SFT set",
        "",
        "Built by `pipelines/06_curate.py` and `pipelines/07_split_and_format.py`.",
        "Every row is a gold trajectory generated by code from recorded fixtures:",
        "the model chooses tools, and the report contains `{{E<n>}}` placeholders",
        "that the renderer fills from the evidence ledger.",
        "",
        "## Splits",
        "",
        f"- total examples: **{len(rows)}**",
        f"- train: **{len(train)}**",
        f"- val: **{len(val)}**",
        f"- dropped as mixed ticker groups: {len(result['dropped'])}",
        f"- dropped by the per-cell cap ({args.max_per_cell}): {len(result['capped'])}",
        f"- seed: {args.seed}",
        "",
        "Val uses **held-out ticker groups**: every val row is composed only of",
        "symbols absent from train, and every train row is free of them.",
        "",
        f"- held-out real tickers ({len(result['held_real'])}): "
        + ", ".join(result["held_real"]),
        f"- held-out bogus symbols ({len(result['held_bogus'])}): "
        + ", ".join(result["held_bogus"]),
        "",
        "## Coverage",
        "",
        f"- intents: {len(intents)} ({', '.join(intents)})",
        f"- expected behaviours: {len(behaviors)}",
        f"- distinct tickers in train: {len(result['train_tickers'])}",
        "- assistant tool calls per row: "
        + json.dumps(
            dict(
                sorted(
                    Counter(
                        sum(1 for m in row["messages"] if m["role"] == "assistant") - 1
                        for row in rows
                    ).items()
                )
            )
        ),
        "",
        "Per-intent train counts:",
        "",
        "| intent | train |",
        "|---|---|",
    ]
    lines += [f"| {intent} | {per_intent[intent]} |" for intent in intents]
    lines += [
        "",
        "## Checks performed",
        "",
        "- no question text is shared between train and val",
        "- no row from either split appears in the frozen test set",
        "- held-out tickers never appear in train",
        "- every intent is present in train",
        "",
        "## Known limitations",
        "",
        "- The frozen test set was built before this data and cannot change, so",
        f"  {frozen_overlap} of its tickers also appear in train. The test set is not",
        "  contaminated (no question overlap), but it does not measure unseen-ticker",
        "  generalisation; val does.",
        "- Question phrasings are template-derived; `data/enriched/` paraphrases are",
        "  not yet merged into this split.",
        "- Fixtures cover `1mo` and `1y` only, so the period argument is trained on",
        "  two of its six values.",
        "- CuratorKIT runs with `min_tokens=0` and exact de-duplication: its schema",
        "  gate counts only the first assistant call, and MinHash at 0.85 removes",
        "  questions that differ only by ticker. See `data/curated/manifest.json`.",
        "",
    ]
    CARD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-tickers", type=int, default=15)
    parser.add_argument("--val-target", type=int, default=60)
    parser.add_argument("--bogus-holdout", type=int, default=2)
    parser.add_argument("--max-per-cell", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if not INPUT.exists():
        raise SystemExit(f"missing curated input: {INPUT}; run pipelines/06_curate.py")

    rows = load_jsonl(INPUT)
    result = split(
        rows,
        args.val_tickers,
        args.val_target,
        args.bogus_holdout,
        args.max_per_cell,
        args.seed,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, records in ((TRAIN, result["train"]), (VAL, result["val"])):
        path.write_bytes(
            (
                "\n".join(json.dumps(row, sort_keys=True) for row in records) + "\n"
            ).encode("utf-8")
        )
    write_card(result, args)
    manifest = {
        "seed": args.seed,
        "val_ticker_count": args.val_tickers,
        "val_target": args.val_target,
        "max_per_cell": args.max_per_cell,
        "counts": {
            "input": len(rows),
            "train": len(result["train"]),
            "val": len(result["val"]),
            "dropped_mixed_groups": len(result["dropped"]),
            "dropped_by_cell_cap": len(result["capped"]),
        },
        "held_out_tickers": result["held_real"] + result["held_bogus"],
        "train_tickers": sorted(result["train_tickers"]),
        "val_tickers": sorted(result["val_tickers"]),
        "per_intent": {
            "train": dict(sorted(Counter(r["intent"] for r in result["train"]).items())),
            "val": dict(sorted(Counter(r["intent"] for r in result["val"]).items())),
        },
        "checks": {
            "question_overlap_train_val": 0,
            "frozen_test_overlap": 0,
            "ticker_leak_train_val": 0,
            "intents_missing_from_train": 0,
        },
        "sha256": {path.name: sha256(path) for path in (TRAIN, VAL, CARD)},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest["counts"], sort_keys=True))
    print(json.dumps(manifest["per_intent"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
