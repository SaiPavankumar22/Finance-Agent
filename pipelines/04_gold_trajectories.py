"""Build deterministic WS4 gold tool-calling trajectories.

This stage uses code, not an LLM, to choose tools, arguments, fixture results,
evidence, and report placeholders. It is intentionally separate from later
Anthropic paraphrasing and CuratorKIT filtering.

Usage:
    python pipelines/02_inject_failures.py --limit 40
    python pipelines/04_gold_trajectories.py --count 800
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evidence.ledger import EvidenceLedger
from app.evidence.render import render_ledger
from app.schemas.report import Claim, MissingItem, ReportSkeleton
from app.schemas.tools import ToolResult
from app.tools.registry import evidence_fields, tool_prompt
from app.tools.replay import DEFAULT_FIXTURES, FixtureStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS_DIR = PROJECT_ROOT / "data" / "seeds"
INJECTED_DIR = PROJECT_ROOT / "data" / "injected"
EVAL_TESTSET = PROJECT_ROOT / "eval" / "testset.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "gold"
OUTPUT_FILE = OUTPUT_DIR / "trajectories.jsonl"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"

PERIOD_WORDS = {
    "1mo": "past month",
    "3mo": "past quarter",
    "6mo": "past six months",
    "1y": "past year",
    "2y": "past two years",
    "5y": "past five years",
}


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    intents = yaml.safe_load((SEEDS_DIR / "intents.yaml").read_text(encoding="utf-8"))[
        "intents"
    ]
    templates = yaml.safe_load(
        (SEEDS_DIR / "question_templates.yaml").read_text(encoding="utf-8")
    )
    tickers = json.loads((SEEDS_DIR / "tickers.json").read_text(encoding="utf-8"))
    return intents, templates, tickers


def load_frozen_questions() -> set[str]:
    if not EVAL_TESTSET.exists():
        return set()
    return {
        json.loads(line)["question"]
        for line in EVAL_TESTSET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_injected() -> dict[str, list[dict[str, Any]]]:
    index_path = INJECTED_DIR / "index.json"
    if not index_path.exists():
        return {}
    result: dict[str, list[dict[str, Any]]] = {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for entry in index.get("entries", []):
        result.setdefault(entry["mutation"], []).append(
            json.loads((INJECTED_DIR / entry["file"]).read_text(encoding="utf-8"))
        )
    return result


def pick_symbols(
    spec: dict[str, Any],
    tickers: dict[str, Any],
    rng: random.Random,
    injected: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], dict[str, Any] | None]:
    mutation = spec.get("mutation", "none")
    if mutation in injected and injected[mutation]:
        snapshot = rng.choice(injected[mutation])
        if mutation == "invalid_ticker":
            return ["ZZZZ"], snapshot
        return list(snapshot["source_tickers"]), snapshot

    pool_name = spec.get("pool", "us")
    if pool_name == "us":
        pool = [x["symbol"] for x in tickers["tickers"] if x["kind"].startswith("us")]
    elif pool_name == "non_us":
        pool = [x["symbol"] for x in tickers["tickers"] if x["kind"] == "non_us"]
    elif pool_name == "bogus":
        return ["ZZZZ"], None
    else:
        pool = [x["symbol"] for x in tickers["tickers"]]
    count = int(spec.get("tickers", 1))
    return (rng.sample(pool, count), None)


def result_for(
    store: FixtureStore,
    snapshot: dict[str, Any] | None,
    tool: str,
    ticker: str,
    period: str,
) -> ToolResult:
    if snapshot is not None:
        key = tool if tool != "get_historical_prices" else f"{tool}@{period}"
        wrapped = snapshot.get("fixtures", {}).get(ticker, {}).get(key)
        if wrapped is not None:
            return ToolResult.model_validate(wrapped["result"])
    lookup_period = period if tool == "get_historical_prices" else None
    return store.result(tool, ticker, lookup_period)


def call_ticker(tool: str, position: int, symbols: list[str]) -> str:
    if len(symbols) == 1:
        return symbols[0]
    # The comparison gold sequence is quote(A), quote(B), ratios(A), ratios(B).
    if tool == "get_stock_quote":
        return symbols[position % len(symbols)]
    return symbols[(position - 2) % len(symbols)]


def build_report(
    question: str,
    ledger: EvidenceLedger,
    results: list[tuple[str, ToolResult]],
    behavior: str,
) -> dict[str, Any]:
    claims = [
        Claim(
            type="fact",
            text=f"The recorded {entry.label} is {{{{{entry.id}}}}}.",
            evidence=[entry.id],
        )
        for entry in ledger
        if entry.value is not None
    ]
    missing = [
        MissingItem(
            item=tool,
            reason=result.error or f"provider returned {result.status}",
        )
        for tool, result in results
        if not result.usable
    ]
    skeleton = ReportSkeleton(
        claims=claims,
        missing=missing,
        limitations=["Values come from a recorded provider snapshot."],
    )
    return {
        "question": question,
        "behavior": behavior,
        "skeleton": skeleton.model_dump(),
        "ledger": ledger.to_dicts(),
        "ledger_text": render_ledger(ledger),
    }


def build_example(
    index: int,
    intent: str,
    spec: dict[str, Any],
    template: str,
    symbols: list[str],
    snapshot: dict[str, Any] | None,
    period: str,
    store: FixtureStore,
) -> dict[str, Any]:
    question = template.format(
        ticker=symbols[0],
        ticker2=symbols[1] if len(symbols) > 1 else "",
        bogus=symbols[0],
        period=PERIOD_WORDS.get(period, period),
    )
    ledger = EvidenceLedger()
    results: list[tuple[str, ToolResult]] = []
    messages: list[dict[str, str]] = [
        {"role": "system", "content": tool_prompt()},
        {"role": "user", "content": question},
    ]
    occurrences: Counter[str] = Counter()
    for step, tool in enumerate(spec["tools"]):
        if tool == "finish":
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps({"tool": "finish", "args": {}}),
                }
            )
            continue
        occurrence = occurrences[tool]
        occurrences[tool] += 1
        ticker = call_ticker(tool, occurrence, symbols)
        args: dict[str, Any] = {"ticker": ticker}
        if tool == "get_historical_prices":
            args["period"] = period
        result = result_for(store, snapshot, tool, ticker, period)
        results.append((tool, result))
        call_id = f"c{step + 1}"
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps({"tool": tool, "args": args}, separators=(",", ":")),
            }
        )
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(
                    {"call_id": call_id, "result": result.model_dump(mode="json")},
                    sort_keys=True,
                ),
            }
        )
        ledger.add_from_tool_result(
            result,
            tool_call_id=call_id,
            fields=evidence_fields(tool, result.data),
        )

    report = build_report(question, ledger, results, spec["expected_behavior"])
    return {
        "id": f"gold_{index:06d}",
        "intent": intent,
        "mutation": spec.get("mutation", "none"),
        "language": "en",
        "tickers": symbols,
        "period": period,
        "question": question,
        "expected_behavior": spec["expected_behavior"],
        "messages": messages,
        "report": report,
    }


def build(count: int, seed: int) -> list[dict[str, Any]]:
    intents, templates, tickers = load_inputs()
    store = FixtureStore(DEFAULT_FIXTURES)
    injected = load_injected()
    frozen_questions = load_frozen_questions()
    periods = store.periods() or ["1y"]
    rng = random.Random(seed)
    names = list(intents)
    examples: list[dict[str, Any]] = []
    used = set(frozen_questions)
    attempts = 0
    while len(examples) < count and attempts < count * 100:
        attempts += 1
        intent = names[(attempts - 1) % len(names)]
        spec = intents[intent]
        symbols, snapshot = pick_symbols(spec, tickers, rng, injected)
        period = rng.choice(spec.get("periods") or periods)
        template = rng.choice(templates[intent]["en"])
        question = template.format(
            ticker=symbols[0],
            ticker2=symbols[1] if len(symbols) > 1 else "",
            bogus=symbols[0],
            period=PERIOD_WORDS.get(period, period),
        )
        if question in used:
            continue
        example = build_example(
            len(examples) + 1,
            intent,
            spec,
            template,
            symbols,
            snapshot,
            period,
            store,
        )
        examples.append(example)
        used.add(question)
    if len(examples) < count:
        raise RuntimeError(f"only generated {len(examples)} of {count} examples")
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    examples = build(args.count, args.seed)
    payload = "\n".join(json.dumps(item, sort_keys=True) for item in examples) + "\n"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_bytes(payload.encode("utf-8"))
    MANIFEST_FILE.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "count": len(examples),
                "sha256": digest,
                "seed": args.seed,
                "frozen_test_questions_excluded": True,
                "intents": dict(sorted(Counter(x["intent"] for x in examples).items())),
                "mutations": dict(sorted(Counter(x["mutation"] for x in examples).items())),
                "source": "recorded fixtures and deterministic planner",
                "next_gate": "CuratorKIT plus split/format stage",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(examples)} gold trajectories to {OUTPUT_FILE}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
