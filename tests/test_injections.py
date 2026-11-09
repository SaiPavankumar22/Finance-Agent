"""Each injected failure must change exactly what it claims, deterministically."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PIPELINES = Path(__file__).resolve().parents[1] / "pipelines"
_spec = importlib.util.spec_from_file_location(
    "inject_failures", PIPELINES / "02_inject_failures.py"
)
injections = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = injections  # dataclasses need the module registered
_spec.loader.exec_module(injections)

NOW = "2026-11-09T12:00:00+00:00"
TOOLS = ("get_stock_quote", "get_financial_ratios", "get_historical_prices@1y", "get_company_news")


def fixture(tool, data, status="ok", error=None):
    return {
        "ticker": "AAPL",
        "tool": tool.split("@")[0],
        "period": tool.split("@")[1] if "@" in tool else None,
        "recorded_at": NOW,
        "result": {
            "status": status,
            "tool": tool.split("@")[0],
            "args": {"ticker": "AAPL"},
            "data": data,
            "source": "yfinance",
            "retrieved_at": NOW,
            "as_of": NOW,
            "error": error,
        },
    }


def base_fixtures():
    return {
        "AAPL": {
            "get_stock_quote": fixture(
                "get_stock_quote",
                {"price": 100.0, "currency": "USD", "volume": 1_000, "market_cap": 1e12},
            ),
            "get_financial_ratios": fixture(
                "get_financial_ratios",
                {"pe_ratio": 20.0, "price_to_book": 5.0, "debt_to_equity": 80.0},
            ),
            "get_historical_prices@1y": fixture(
                "get_historical_prices@1y", {"observations": 250, "sma_20": 99.0}
            ),
            "get_company_news": fixture(
                "get_company_news", {"count": 1, "items": [{"title": "A"}]}
            ),
        },
        "SAP.DE": {
            "get_stock_quote": fixture(
                "get_stock_quote", {"price": 200.0, "currency": "EUR", "volume": 2_000}
            ),
            "get_financial_ratios": fixture("get_financial_ratios", {"pe_ratio": 30.0}),
        },
    }


def flatten(value, prefix=""):
    flat = {}
    if isinstance(value, dict):
        for key, item in value.items():
            flat.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flat.update(flatten(item, f"{prefix}.{index}"))
    else:
        flat[prefix] = value
    return flat


def changed_paths(before, after):
    left, right = flatten(before), flatten(after)
    return {path for path in set(left) | set(right) if left.get(path) != right.get(path)}


def test_mutations_only_touch_their_own_paths():
    fixtures = base_fixtures()
    tickers = ("AAPL", "SAP.DE")
    allowed = {
        "null_ratio": tuple(f"{t}.get_financial_ratios.result.data." for t in tickers),
        "news_unavailable": ("AAPL.get_company_news.result.",),
        "stale_quote": tuple(f"{t}.get_stock_quote.result." for t in tickers),
        # Error fixtures are rebuilt wholesale, so wrapper metadata changes too.
        "provider_error": tuple(f"{t}.get_stock_quote." for t in tickers),
        "malformed": tuple(f"{t}.get_financial_ratios." for t in tickers),
        "currency_mismatch": ("SAP.DE.get_stock_quote.result.data.",),
        "unit_mismatch": tuple(f"{t}.get_stock_quote.result.data." for t in tickers),
        "conflict": (),
        "invalid_ticker": tuple(f"{t}." for t in (*tickers, "ZZZZ")),
        "loop_trap": tuple(f"{t}.{tool}." for t in tickers for tool in TOOLS),
    }
    for name in injections.MUTATIONS:
        before = copy.deepcopy(fixtures)
        mutated = injections.apply_mutation(fixtures, name, seed=3)
        changed = changed_paths(fixtures, mutated.fixtures)
        if name == "conflict":
            assert changed == set(), "conflict only annotates; it must not rewrite fixtures"
            continue
        assert changed, f"{name} changed nothing"
        assert all(path.startswith(allowed[name]) for path in changed), (
            name,
            sorted(changed),
        )
        assert fixtures == before, f"{name} mutated its input"


def test_null_ratio_nulls_every_ratio_field():
    mutated = injections.apply_mutation(base_fixtures(), "null_ratio")
    data = mutated.fixtures["AAPL"]["get_financial_ratios"]["result"]["data"]
    assert all(value is None for value in data.values())
    assert mutated.expected_behavior == "report_missing_ratios"


def test_news_unavailable_is_an_explicit_status():
    mutated = injections.apply_mutation(base_fixtures(), "news_unavailable")
    result = mutated.fixtures["AAPL"]["get_company_news"]["result"]
    assert result["status"] == "unavailable"
    assert result["data"] is None
    assert mutated.expected_behavior == "report_news_unavailable"


def test_stale_quote_pushes_as_of_past_the_threshold():
    from datetime import datetime, timezone

    mutated = injections.apply_mutation(base_fixtures(), "stale_quote")
    result = mutated.fixtures["AAPL"]["get_stock_quote"]["result"]
    assert result["status"] == "stale"
    as_of = datetime.fromisoformat(result["as_of"])
    assert (datetime.now(timezone.utc) - as_of).days > 300


def test_provider_error_keeps_data_out():
    mutated = injections.apply_mutation(base_fixtures(), "provider_error")
    result = mutated.fixtures["AAPL"]["get_stock_quote"]["result"]
    assert result["status"] == "error"
    assert result["data"] is None
    assert result["error"] == "provider_unavailable"


def test_malformed_carries_a_truncated_payload_annotation():
    mutated = injections.apply_mutation(base_fixtures(), "malformed")
    ratios_result = mutated.fixtures["AAPL"]["get_financial_ratios"]["result"]
    assert ratios_result["error"] == "malformed_response"
    assert mutated.annotations["malformed_payload"].endswith(":")
    assert mutated.expected_behavior == "abstain_malformed_response"


def test_currency_mismatch_only_rewrites_the_partner():
    mutated = injections.apply_mutation(base_fixtures(), "currency_mismatch")
    data = mutated.fixtures["SAP.DE"]["get_stock_quote"]["result"]["data"]
    assert data["currency"] == "EUR"
    assert mutated.fixtures["AAPL"]["get_stock_quote"]["result"]["data"]["currency"] == "USD"
    assert mutated.annotations["mismatch_ticker"] == "SAP.DE"


def test_currency_mismatch_needs_two_tickers():
    single = {"AAPL": base_fixtures()["AAPL"]}
    with pytest.raises(ValueError):
        injections.apply_mutation(single, "currency_mismatch")


def test_unit_mismatch_records_the_scale():
    mutated = injections.apply_mutation(base_fixtures(), "unit_mismatch")
    data = mutated.fixtures["AAPL"]["get_stock_quote"]["result"]["data"]
    assert data["unit_annotations"] == {"volume": "thousands"}
    assert data["volume"] == 1_000_000


def test_conflict_reports_a_value_beyond_tolerance():
    mutated = injections.apply_mutation(base_fixtures(), "conflict", seed=11)
    conflict = mutated.annotations["conflict"]
    assert conflict["ticker"] == "AAPL"
    gap = abs(conflict["sec_edgar"] - conflict["yfinance"]) / conflict["yfinance"]
    assert gap > conflict["tolerance"]
    assert mutated.expected_behavior == "surface_conflict"


def test_invalid_ticker_replaces_every_tool_with_an_error():
    mutated = injections.apply_mutation(base_fixtures(), "invalid_ticker")
    assert list(mutated.fixtures) == ["ZZZZ"]
    statuses = {
        fixture["result"]["status"] for fixture in mutated.fixtures["ZZZZ"].values()
    }
    errors = {fixture["result"]["error"] for fixture in mutated.fixtures["ZZZZ"].values()}
    assert statuses == {"error"}
    assert errors == {"invalid_ticker"}


def test_loop_trap_makes_every_tool_fail_identically():
    mutated = injections.apply_mutation(base_fixtures(), "loop_trap")
    errors = {
        fixture["result"]["error"]
        for ticker in mutated.fixtures.values()
        for fixture in ticker.values()
    }
    assert errors == {"insufficient_data"}
    assert mutated.expected_behavior == "stop_after_bounded_attempts"


def test_mutations_are_deterministic_given_a_seed():
    first = injections.apply_mutation(base_fixtures(), "conflict", seed=5)
    second = injections.apply_mutation(base_fixtures(), "conflict", seed=5)
    assert json.dumps(first.annotations, sort_keys=True) == json.dumps(
        second.annotations, sort_keys=True
    )


def test_unknown_mutation_is_an_error():
    with pytest.raises(KeyError):
        injections.apply_mutation(base_fixtures(), "earthquake")
