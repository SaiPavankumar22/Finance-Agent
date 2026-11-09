"""Build and freeze the evaluation test set (plan sections 11.2-11.3, step C).

The test set is generated **before any training data exists** and frozen with a
SHA-256 hash. Regenerating it requires ``--force`` and changes the hash, which is
the point: a result measured against a moving test set means nothing.

Usage:
    python pipelines/03_make_seed_questions.py
    python pipelines/03_make_seed_questions.py --force --count 150
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools.replay import DEFAULT_FIXTURES, FixtureStore  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS_DIR = PROJECT_ROOT / "data" / "seeds"
EVAL_DIR = PROJECT_ROOT / "eval"
TESTSET = EVAL_DIR / "testset.jsonl"
HASH_FILE = EVAL_DIR / "testset.sha256"
META_FILE = EVAL_DIR / "testset.meta.json"

PERIOD_WORDS = {
    "1mo": "past month",
    "3mo": "past quarter",
    "6mo": "past six months",
    "1y": "past year",
    "2y": "past two years",
    "5y": "past five years",
}
USABLE = frozenset({"ok", "stale"})
MIN_EXAMPLES = 120
#: Attempts to find an eligible, not-yet-used (ticker, template) combination.
PICK_ATTEMPTS = 60


def load_seeds() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    intents = yaml.safe_load((SEEDS_DIR / "intents.yaml").read_text(encoding="utf-8"))["intents"]
    templates = yaml.safe_load(
        (SEEDS_DIR / "question_templates.yaml").read_text(encoding="utf-8")
    )
    tickers = json.loads((SEEDS_DIR / "tickers.json").read_text(encoding="utf-8"))
    return intents, templates, tickers


def allocate(intents: dict[str, Any], count: int) -> dict[str, int]:
    """Largest-remainder allocation of ``count`` examples across intent shares."""

    total_share = sum(spec["share"] for spec in intents.values())
    raw = {name: spec["share"] / total_share * count for name, spec in intents.items()}
    quotas = {name: int(value) for name, value in raw.items()}
    remainder = count - sum(quotas.values())
    for name in sorted(raw, key=lambda key: raw[key] - quotas[key], reverse=True):
        if remainder <= 0:
            break
        quotas[name] += 1
        remainder -= 1
    return quotas


def pick_pool(spec: dict[str, Any], tickers: dict[str, Any]) -> list[str]:
    pool = spec.get("pool", "us")
    if pool == "bogus":
        return list(tickers["bogus"])
    entries = tickers["tickers"]
    if pool == "us":
        return [entry["symbol"] for entry in entries if entry["kind"].startswith("us")]
    if pool == "non_us":
        return [entry["symbol"] for entry in entries if entry["kind"] == "non_us"]
    return [entry["symbol"] for entry in entries]


def fixture_ok(store: FixtureStore, tool: str, ticker: str, period: str | None) -> bool:
    lookup_period = period if tool == "get_historical_prices" else None
    if not store.has(tool, ticker, lookup_period):
        return False
    return store.result(tool, ticker, lookup_period).status in USABLE


def eligible(store: FixtureStore, spec: dict[str, Any], chosen: list[str], period: str) -> bool:
    if not chosen:
        return False
    for tool in spec.get("needs", []):
        for ticker in chosen:
            lookup_period = period if tool == "get_historical_prices" else None
            if not store.has(tool, ticker, lookup_period):
                return False
    for tool in spec.get("requires_ok", []):
        for ticker in chosen:
            if not fixture_ok(store, tool, ticker, period):
                return False
    return True


#: English only, by project decision: the multilingual P1 item is dropped, so
#: the test set and the training data stay in one language.
TEST_LANGUAGE = "en"


def build_example(
    intent: str,
    spec: dict[str, Any],
    rng: random.Random,
    store: FixtureStore,
    tickers: dict[str, Any],
    variants: dict[str, list[str]],
    periods: list[str],
    index: int,
    used_questions: set[str],
) -> dict[str, Any] | None:
    pool = pick_pool(spec, tickers)
    allowed_periods = spec.get("periods") or periods or ["1y"]

    for _ in range(PICK_ATTEMPTS):
        count = spec.get("tickers", 1)
        chosen = [rng.choice(pool)] if count == 1 else rng.sample(pool, count)
        period = rng.choice(allowed_periods)
        if not eligible(store, spec, chosen, period):
            continue

        template = rng.choice(variants[TEST_LANGUAGE]).strip()
        question = template.format(
            ticker=chosen[0],
            ticker2=chosen[1] if len(chosen) > 1 else "",
            bogus=chosen[0],
            period=PERIOD_WORDS.get(period, period),
        )
        if question in used_questions:
            continue

        statuses: dict[str, str] = {}
        for tool in spec["tools"]:
            # Bogus-symbol intents have no fixtures by design: the failure is the point.
            if tool == "finish" or spec.get("pool") == "bogus":
                continue
            for ticker in chosen:
                lookup_period = period if tool == "get_historical_prices" else None
                statuses[f"{tool}@{ticker}"] = store.result(tool, ticker, lookup_period).status

        return {
            "id": f"ts_{index:04d}",
            "intent": intent,
            "language": TEST_LANGUAGE,
            "tickers": chosen,
            "period": period,
            "mutation": spec.get("mutation", "none"),
            "expected_behavior": spec["expected_behavior"],
            "question": question,
            "gold_tools": list(spec["tools"]),
            "expect_abstain": bool(spec.get("abstain", False)),
            "fixture_status": statuses,
        }
    return None


def build(count: int, seed: int) -> list[dict[str, Any]]:
    intents, templates, tickers = load_seeds()
    store = FixtureStore(DEFAULT_FIXTURES)
    periods = store.periods() or ["1y"]
    rng = random.Random(seed)
    quotas = allocate(intents, count)

    examples: list[dict[str, Any]] = []
    used_questions: set[str] = set()
    index = 1

    def attempt(intent: str) -> bool:
        nonlocal index
        spec = intents[intent]
        example = build_example(
            intent,
            spec,
            rng,
            store,
            tickers,
            templates[intent],
            periods,
            index,
            used_questions,
        )
        if example is None:
            return False
        examples.append(example)
        used_questions.add(example["question"])
        index += 1
        return True

    for intent in intents:
        filled = 0
        for _ in range(quotas[intent]):
            if not attempt(intent):
                print(f"warning: could not fill quota for {intent} ({filled} made)")
                break
            filled += 1

    attempts = 0
    while len(examples) < count and attempts < count * 4:
        attempts += 1
        progressed = False
        for intent in intents:
            if len(examples) >= count:
                break
            progressed = attempt(intent) or progressed
        if not progressed:
            break

    _assert_unique(examples)
    return examples


def _assert_unique(examples: list[dict[str, Any]]) -> None:
    ids = Counter(example["id"] for example in examples)
    questions = Counter(example["question"] for example in examples)
    if any(value > 1 for value in ids.values()):
        raise SystemExit("duplicate example ids generated")
    if any(value > 1 for value in questions.values()):
        raise SystemExit("duplicate questions generated")


def serialise(examples: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(example, sort_keys=True) for example in examples) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    examples = build(args.count, args.seed)
    payload = serialise(examples)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    if TESTSET.exists() and not args.force:
        existing = HASH_FILE.read_text(encoding="utf-8").split()[0] if HASH_FILE.exists() else ""
        if existing != digest:
            print(
                "refusing to overwrite the frozen test set: generated content differs.\n"
                f"  frozen:    {existing}\n  generated: {digest}\n"
                "Pass --force only if you intend to invalidate every recorded baseline."
            )
            return 1
        print(f"test set already frozen at {digest}")
        return 0

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    # Bytes, not text mode: Windows newline translation would change the hash.
    TESTSET.write_bytes(payload.encode("utf-8"))
    HASH_FILE.write_bytes(f"{digest}  testset.jsonl\n".encode("utf-8"))

    store = FixtureStore(DEFAULT_FIXTURES)
    intent_counts = Counter(example["intent"] for example in examples)
    language_counts = Counter(example["language"] for example in examples)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": len(examples),
        "sha256": digest,
        "seed": args.seed,
        "target_count": args.count,
        "fixture_snapshot": store.manifest().get("snapshot_date"),
        "fixture_hash": store.manifest().get("content_hash"),
        "fixture_count": store.count(),
        "fixture_tickers": len(store.tickers()),
        "intent_counts": dict(sorted(intent_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "abstain_count": sum(1 for example in examples if example["expect_abstain"]),
        "notes": [
            "Built before any training data existed (plan WS2).",
            "conflict intent is excluded until the SEC EDGAR provider lands (P1).",
            "overview examples may report news as unavailable: that is provider truth.",
            "question paraphrasing by an LLM happens later, for training data only.",
            "English only: the multilingual variants were dropped by project decision.",
        ],
    }
    META_FILE.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"frozen {len(examples)} examples at {digest}")
    print(f"intents: {dict(sorted(intent_counts.items()))}")
    print(f"languages: {dict(sorted(language_counts.items()))}")
    if len(examples) < MIN_EXAMPLES:
        print(f"only {len(examples)} examples generated; expected at least {MIN_EXAMPLES}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
