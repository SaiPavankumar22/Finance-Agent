"""Inject deliberate failures into recorded fixtures (plan section 11.3, step B).

Every mutation is a named function with a known correct behaviour, so the gold
answer is computable without an LLM. Mutations are deterministic given a seed and
never modify their input.

Fixture set shape: ``{ticker: {tool: ToolResult-as-dict}}``.

Usage:
    python pipelines/02_inject_failures.py --limit 20
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools.replay import DEFAULT_FIXTURES, FixtureStore  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INJECTED_DIR = PROJECT_ROOT / "data" / "injected"
TOOLS: tuple[str, ...] = (
    "get_stock_quote",
    "get_financial_ratios",
    "get_historical_prices",
    "get_company_news",
)
RATIO_FIELDS: tuple[str, ...] = (
    "pe_ratio",
    "price_to_book",
    "enterprise_value_to_ebitda",
    "debt_to_equity",
    "return_on_equity",
)

FixtureSet = dict[str, dict[str, dict[str, Any]]]


@dataclass
class MutationResult:
    """A mutated fixture set plus the behaviour it should provoke."""

    fixtures: FixtureSet
    expected_behavior: str
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Mutation:
    name: str
    expected_behavior: str
    description: str
    fn: Callable[[FixtureSet, random.Random], MutationResult]
    tickers_needed: int = 1


def _wrap(tool_key: str, symbol: str, result: dict[str, Any]) -> dict[str, Any]:
    """Fixture-wrapper shape: metadata around a serialised ToolResult."""

    tool, _, period = tool_key.partition("@")
    return {
        "ticker": symbol,
        "tool": tool,
        "period": period or None,
        "recorded_at": result.get("retrieved_at"),
        "result": result,
    }


def _store(store: FixtureStore, tickers: list[str]) -> FixtureSet:
    fixtures: FixtureSet = {}
    for symbol in tickers:
        fixtures[symbol] = {}
        for tool in TOOLS:
            periods = store.periods() or ["1y"]
            if tool == "get_historical_prices":
                for period in periods:
                    key = f"{tool}@{period}"
                    fixtures[symbol][key] = _wrap(
                        key, symbol, store.result(tool, symbol, period).model_dump(mode="json")
                    )
            else:
                fixtures[symbol][tool] = _wrap(
                    tool, symbol, store.result(tool, symbol).model_dump(mode="json")
                )
    return fixtures


def _result(fixture: dict[str, Any]) -> dict[str, Any]:
    return fixture["result"]


def _error_fixture(tool: str, symbol: str, code: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "ticker": symbol,
        "tool": tool.split("@")[0],
        "period": tool.split("@")[1] if "@" in tool else None,
        "recorded_at": now,
        "result": {
            "status": "error",
            "tool": tool.split("@")[0],
            "args": {"ticker": symbol},
            "data": None,
            "source": "yfinance",
            "retrieved_at": now,
            "as_of": None,
            "error": code,
        },
    }


def _null_ratio(fixtures: FixtureSet, rng: random.Random) -> MutationResult:
    mutated = copy.deepcopy(fixtures)
    for tools in mutated.values():
        for key, fixture in tools.items():
            if key.startswith("get_financial_ratios"):
                data = _result(fixture).get("data") or {}
                for ratio in RATIO_FIELDS:
                    data[ratio] = None
                _result(fixture)["data"] = data
    return MutationResult(mutated, "report_missing_ratios")


def _news_unavailable(fixtures: FixtureSet, rng: random.Random) -> MutationResult:
    mutated = copy.deepcopy(fixtures)
    for tools in mutated.values():
        for key, fixture in tools.items():
            if key.startswith("get_company_news"):
                _result(fixture).update(
                    {"status": "unavailable", "data": None, "error": "provider_unavailable"}
                )
    return MutationResult(mutated, "report_news_unavailable")


def _stale_quote(fixtures: FixtureSet, rng: random.Random) -> MutationResult:
    mutated = copy.deepcopy(fixtures)
    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    for tools in mutated.values():
        for key, fixture in tools.items():
            if key.startswith("get_stock_quote"):
                _result(fixture).update({"status": "stale", "as_of": old})
    return MutationResult(mutated, "mark_stale_data")


def _provider_error(fixtures: FixtureSet, rng: random.Random) -> MutationResult:
    mutated = copy.deepcopy(fixtures)
    for symbol, tools in mutated.items():
        for key in list(tools):
            if key.startswith("get_stock_quote"):
                tools[key] = _error_fixture(key, symbol, "provider_unavailable")
    return MutationResult(mutated, "abstain_provider_error")


def _malformed(fixtures: FixtureSet, rng: random.Random) -> MutationResult:
    mutated = copy.deepcopy(fixtures)
    for symbol, tools in mutated.items():
        for key in list(tools):
            if key.startswith("get_financial_ratios"):
                tools[key] = _error_fixture(key, symbol, "malformed_response")
    annotations = {"malformed_payload": '{"pe_ratio": 28.4, "price_to_book":'}
    return MutationResult(mutated, "abstain_malformed_response", annotations)


def _currency_mismatch(fixtures: FixtureSet, rng: random.Random) -> MutationResult:
    mutated = copy.deepcopy(fixtures)
    symbols = list(mutated)
    if len(symbols) < 2:
        raise ValueError("currency_mismatch needs two tickers")
    other = symbols[1]
    for key, fixture in mutated[other].items():
        if key.startswith("get_stock_quote"):
            data = _result(fixture).get("data") or {}
            data["currency"] = "EUR"
            if isinstance(data.get("price"), (int, float)):
                data["price"] = round(float(data["price"]) * 0.9, 2)
            _result(fixture)["data"] = data
    return MutationResult(mutated, "flag_currency_mismatch", {"mismatch_ticker": other})


def _unit_mismatch(fixtures: FixtureSet, rng: random.Random) -> MutationResult:
    mutated = copy.deepcopy(fixtures)
    for tools in mutated.values():
        for key, fixture in tools.items():
            if key.startswith("get_stock_quote"):
                data = _result(fixture).get("data") or {}
                volume = data.get("volume")
                if isinstance(volume, (int, float)):
                    data["volume"] = int(volume) * 1000
                data["unit_annotations"] = {"volume": "thousands"}
                _result(fixture)["data"] = data
    return MutationResult(mutated, "flag_unit_mismatch")


def _conflict(fixtures: FixtureSet, rng: random.Random) -> MutationResult:
    mutated = copy.deepcopy(fixtures)
    symbol = next(iter(mutated))
    ratios = _result(mutated[symbol]["get_financial_ratios"]).get("data") or {}
    recorded = ratios.get("pe_ratio")
    alternative = round(float(recorded) * (1.0 + rng.uniform(0.15, 0.35)), 2) if recorded else 99.0
    annotations = {
        "conflict": {
            "ticker": symbol,
            "field": "pe_ratio",
            "yfinance": recorded,
            "sec_edgar": alternative,
            "tolerance": 0.05,
        }
    }
    return MutationResult(mutated, "surface_conflict", annotations)


def _invalid_ticker(fixtures: FixtureSet, rng: random.Random) -> MutationResult:
    symbol = next(iter(fixtures))
    bogus = "ZZZZ"
    mutated: FixtureSet = {bogus: {}}
    for tool in fixtures[symbol]:
        mutated[bogus][tool] = _error_fixture(tool, bogus, "invalid_ticker")
    return MutationResult(mutated, "abstain_invalid_ticker", {"replaced_ticker": symbol})


def _loop_trap(fixtures: FixtureSet, rng: random.Random) -> MutationResult:
    mutated = copy.deepcopy(fixtures)
    for symbol, tools in mutated.items():
        for key in list(tools):
            tools[key] = _error_fixture(key, symbol, "insufficient_data")
    return MutationResult(mutated, "stop_after_bounded_attempts")


MUTATIONS: dict[str, Mutation] = {
    "null_ratio": Mutation(
        "null_ratio", "report_missing_ratios", "ratio fields become null", _null_ratio
    ),
    "news_unavailable": Mutation(
        "news_unavailable",
        "report_news_unavailable",
        "news provider returns nothing",
        _news_unavailable,
    ),
    "stale_quote": Mutation(
        "stale_quote", "mark_stale_data", "quote as_of pushed past the threshold", _stale_quote
    ),
    "provider_error": Mutation(
        "provider_error",
        "abstain_provider_error",
        "quote provider errors out",
        _provider_error,
    ),
    "malformed": Mutation(
        "malformed", "abstain_malformed_response", "truncated ratio payload", _malformed
    ),
    "currency_mismatch": Mutation(
        "currency_mismatch",
        "flag_currency_mismatch",
        "two listings in different currencies",
        _currency_mismatch,
        tickers_needed=2,
    ),
    "unit_mismatch": Mutation(
        "unit_mismatch", "flag_unit_mismatch", "volume reported in thousands", _unit_mismatch
    ),
    "conflict": Mutation(
        "conflict", "surface_conflict", "SEC value disagrees with the quote provider", _conflict
    ),
    "invalid_ticker": Mutation(
        "invalid_ticker", "abstain_invalid_ticker", "symbol resolves to nothing", _invalid_ticker
    ),
    "loop_trap": Mutation(
        "loop_trap",
        "stop_after_bounded_attempts",
        "provider repeats the same unhelpful error",
        _loop_trap,
    ),
}


def apply_mutation(
    fixtures: FixtureSet, name: str, *, seed: int = 0
) -> MutationResult:
    """Apply one named mutation deterministically, leaving the input untouched."""

    if name not in MUTATIONS:
        raise KeyError(f"unknown mutation: {name}")
    return MUTATIONS[name].fn(fixtures, random.Random(seed))


def write_injected(
    ticker_key: str, mutation: str, result: MutationResult, seed: int, source: list[str]
) -> Path:
    INJECTED_DIR.mkdir(parents=True, exist_ok=True)
    path = INJECTED_DIR / f"{ticker_key}__{mutation}.json"
    payload = {
        "source_tickers": source,
        "mutation": mutation,
        "expected_behavior": result.expected_behavior,
        "seed": seed,
        "annotations": result.annotations,
        "fixtures": result.fixtures,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="source fixture sets to mutate")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mutations", default="all")
    args = parser.parse_args()

    store = FixtureStore(DEFAULT_FIXTURES)
    tickers = store.tickers()[: args.limit]
    if not tickers:
        print("no fixtures available; run 01_record_fixtures.py first")
        return 1

    names = (
        list(MUTATIONS)
        if args.mutations == "all"
        else [name.strip() for name in args.mutations.split(",")]
    )

    INJECTED_DIR.mkdir(parents=True, exist_ok=True)
    for stale in INJECTED_DIR.glob("*.json"):
        stale.unlink()

    written = 0
    index: list[dict[str, Any]] = []
    for position, symbol in enumerate(tickers):
        for name in names:
            mutation = MUTATIONS[name]
            sources = [symbol]
            if mutation.tickers_needed == 2:
                partner = tickers[(position + 1) % len(tickers)]
                if partner == symbol:
                    continue
                sources = [symbol, partner]
            fixtures = _store(store, sources)
            try:
                result = apply_mutation(fixtures, name, seed=args.seed)
            except (ValueError, KeyError) as error:
                print(f"skip {symbol} {name}: {error}")
                continue
            key = "-".join(sources)
            path = write_injected(key, name, result, args.seed, sources)
            index.append(
                {
                    "file": path.name,
                    "source_tickers": sources,
                    "mutation": name,
                    "expected_behavior": result.expected_behavior,
                }
            )
            written += 1

    (INJECTED_DIR / "index.json").write_text(
        json.dumps({"count": written, "entries": index}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {written} injected fixture sets to {INJECTED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
