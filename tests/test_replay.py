"""Fixture replay must be offline, deterministic and explicit about missing data."""

import json
from datetime import datetime

from app.schemas.tools import ToolResult
from app.tools.replay import EPOCH, FixtureStore, fixture_filename

NOW = "2026-11-09T12:00:00+00:00"


def result_payload(tool, status="ok", data=None, error=None, as_of=NOW):
    return {
        "status": status,
        "tool": tool,
        "args": {"ticker": "AAPL"},
        "data": data,
        "source": "yfinance",
        "retrieved_at": NOW,
        "as_of": as_of,
        "error": error,
    }


def seed_dir(tmp_path):
    payload = result_payload("get_stock_quote", data={"price": 189.23, "currency": "USD"})
    (tmp_path / fixture_filename("get_stock_quote", "AAPL")).write_text(
        json.dumps(
            {"ticker": "AAPL", "tool": "get_stock_quote", "period": None, "result": payload}
        ),
        encoding="utf-8",
    )
    history = result_payload("get_historical_prices", data={"observations": 250})
    (tmp_path / fixture_filename("get_historical_prices", "AAPL", "1y")).write_text(
        json.dumps(
            {
                "ticker": "AAPL",
                "tool": "get_historical_prices",
                "period": "1y",
                "result": history,
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "fixture_count": 2,
        "tickers": ["AAPL"],
        "periods": ["1y"],
        "entries": [
            {"ticker": "AAPL", "tool": "get_stock_quote", "period": None},
            {"ticker": "AAPL", "tool": "get_historical_prices", "period": "1y"},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return FixtureStore(tmp_path)


def test_fixture_filenames_encode_tool_and_period():
    assert fixture_filename("get_stock_quote", "AAPL") == "AAPL_get_stock_quote.json"
    assert (
        fixture_filename("get_historical_prices", "SAP.DE", "1y")
        == "SAP.DE_get_historical_prices_1y.json"
    )


def test_recorded_result_is_returned_as_a_valid_envelope(tmp_path):
    store = seed_dir(tmp_path)
    result = store.result("get_stock_quote", "AAPL")
    assert isinstance(result, ToolResult)
    assert result.status == "ok"
    assert result.data["price"] == 189.23


def test_missing_fixture_is_explicit_not_a_network_call(tmp_path):
    store = seed_dir(tmp_path)
    result = store.result("get_company_news", "AAPL")
    assert result.status == "unavailable"
    assert result.error == "fixture_missing"
    assert result.retrieved_at == EPOCH


def test_has_and_manifest_helpers(tmp_path):
    store = seed_dir(tmp_path)
    assert store.has("get_stock_quote", "AAPL") is True
    assert store.has("get_historical_prices", "AAPL", "1y") is True
    assert store.has("get_historical_prices", "AAPL", "5y") is False
    assert store.tickers() == ["AAPL"]
    assert store.periods() == ["1y"]
    assert store.count() == 2


def test_replay_is_deterministic(tmp_path):
    store = seed_dir(tmp_path)
    first = store.result("get_stock_quote", "AAPL").model_dump(mode="json")
    second = store.result("get_stock_quote", "AAPL").model_dump(mode="json")
    assert first == second


def test_empty_store_returns_synthetic_unavailable(tmp_path):
    store = FixtureStore(tmp_path)
    assert store.count() == 0
    result = store.result("get_stock_quote", "AAPL")
    assert result.error == "fixture_missing"
    assert datetime.fromisoformat(result.retrieved_at.isoformat()) == EPOCH
