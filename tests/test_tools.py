"""Tool layer: never raise, always classify, cache, retry, go stale, trip the breaker."""

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.tools import yfinance_tools as yt

NOW = datetime.now(timezone.utc)


def info(**overrides):
    payload = {
        "symbol": "AAPL",
        "currency": "USD",
        "regularMarketPrice": 189.23,
        "regularMarketVolume": 44_703_806,
        "marketCap": 2.9e12,
        "fiftyTwoWeekLow": 120.0,
        "fiftyTwoWeekHigh": 220.0,
        "regularMarketTime": int(NOW.timestamp()),
        "trailingPE": 28.4,
        "priceToBook": 44.3,
        "enterpriseToEbitda": 29.2,
        "debtToEquity": 84.2,
        "returnOnEquity": 1.42,
    }
    payload.update(overrides)
    return payload


def frame(rows):
    index = pd.date_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=rows, freq="D")
    return pd.DataFrame({"Close": [100.0 + value for value in range(rows)]}, index=index)


def news(count=1):
    return [
        {
            "content": {
                "title": f"Headline {index + 1}",
                "pubDate": f"2026-11-0{index + 1}T10:00:00Z",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://example.com/story"},
            }
        }
        for index in range(count)
    ]


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    yt.reset_state()
    monkeypatch.setattr(yt, "RETRY_BACKOFF_SECONDS", 0.0)
    yield
    yt.reset_state()


def test_quote_success_normalises_fields(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_info", lambda symbol: info())
    result = yt.get_stock_quote("aapl")
    assert result.status == "ok"
    assert result.args == {"ticker": "AAPL"}
    assert result.data["price"] == 189.23
    assert result.data["currency"] == "USD"
    assert result.data["market_cap"] == 2.9e12
    assert result.as_of is not None


def test_results_are_cached_within_the_ttl(monkeypatch):
    calls = []

    def fetch(symbol):
        calls.append(symbol)
        return info()

    monkeypatch.setattr(yt, "_fetch_info", fetch)
    first = yt.get_stock_quote("AAPL")
    second = yt.get_stock_quote("AAPL")
    assert len(calls) == 1
    assert first is second


def test_retries_then_succeeds(monkeypatch):
    calls = []

    def flaky(symbol):
        calls.append(symbol)
        if len(calls) < 3:
            raise ConnectionError("temporary")
        return info()

    monkeypatch.setattr(yt, "_fetch_info", flaky)
    result = yt.get_stock_quote("AAPL")
    assert result.status == "ok"
    assert len(calls) == 3


def test_timeout_becomes_unavailable(monkeypatch):
    attempts = []

    def slow(symbol):
        attempts.append(symbol)
        time.sleep(0.5)
        return info()

    monkeypatch.setattr(yt, "_fetch_info", slow)
    monkeypatch.setattr(yt, "CALL_TIMEOUT_SECONDS", 0.05)
    result = yt.get_stock_quote("AAPL")
    assert result.status == "unavailable"
    assert result.error == "provider_timeout"
    assert len(attempts) == 3


def test_invalid_symbol_pattern_never_reaches_the_provider(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_info", lambda symbol: pytest.fail("provider was called"))
    result = yt.get_stock_quote("!!!")
    assert result.status == "error"
    assert result.error == "invalid_ticker"


def test_empty_payload_is_an_invalid_ticker(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_info", lambda symbol: {})
    result = yt.get_stock_quote("ZZZZ")
    assert result.status == "error"
    assert result.error == "invalid_ticker"


def test_malformed_payload_is_classified_not_raised(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_info", lambda symbol: "not a mapping")
    result = yt.get_stock_quote("AAPL")
    assert result.status == "error"
    assert result.error == "malformed_response"


def test_unparseable_numbers_become_null_not_zero(monkeypatch):
    monkeypatch.setattr(
        yt, "_fetch_info", lambda symbol: info(regularMarketPrice="n/a", marketCap=None)
    )
    result = yt.get_stock_quote("AAPL")
    assert result.status == "ok"
    assert result.data["price"] is None
    assert result.data["market_cap"] is None


def test_old_quote_is_marked_stale(monkeypatch):
    old = int((NOW - timedelta(days=30)).timestamp())
    monkeypatch.setattr(yt, "_fetch_info", lambda symbol: info(regularMarketTime=old))
    result = yt.get_stock_quote("AAPL")
    assert result.status == "stale"


def test_missing_ratios_stay_null(monkeypatch):
    monkeypatch.setattr(
        yt, "_fetch_info", lambda symbol: info(trailingPE=None, forwardPE=None)
    )
    result = yt.get_financial_ratios("AAPL")
    assert result.status == "ok"
    assert result.data["pe_ratio"] is None
    assert result.data["price_to_book"] == 44.3


def test_history_computes_moving_averages(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_history", lambda symbol, period: frame(60))
    result = yt.get_historical_prices("AAPL", "1y")
    assert result.status == "ok"
    assert result.data["observations"] == 60
    assert result.data["sma_20"] is not None
    assert result.data["sma_50"] is not None
    assert result.data["annualized_volatility"] is not None


def test_short_history_has_null_long_average(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_history", lambda symbol, period: frame(10))
    result = yt.get_historical_prices("AAPL", "1mo")
    assert result.data["sma_50"] is None
    assert result.data["sma_20"] is None


def test_empty_history_is_an_invalid_ticker(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_history", lambda symbol, period: frame(0))
    result = yt.get_historical_prices("ZZZZ", "1y")
    assert result.status == "error"
    assert result.error == "invalid_ticker"


def test_period_outside_the_enum_is_refused(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_history", lambda symbol, period: pytest.fail("no fetch"))
    result = yt.get_historical_prices("AAPL", "10y")
    assert result.status == "error"
    assert result.error == "invalid_period"


def test_news_items_are_normalised(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_news", lambda symbol: news(2))
    result = yt.get_company_news("AAPL", 5)
    assert result.status == "ok"
    assert result.data["count"] == 2
    assert result.data["items"][0]["publisher"] == "Reuters"
    assert result.as_of is not None


def test_empty_news_is_explicitly_unavailable(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_news", lambda symbol: [])
    result = yt.get_company_news("AAPL")
    assert result.status == "unavailable"
    assert result.error == "no_news_found"


def test_news_limit_is_validated(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_news", lambda symbol: pytest.fail("no fetch"))
    result = yt.get_company_news("AAPL", 0)
    assert result.status == "error"
    assert result.error == "invalid_limit"


def test_breaker_opens_after_repeated_outages(monkeypatch):
    calls = []

    def boom(symbol):
        calls.append(symbol)
        raise ConnectionError("provider down")

    monkeypatch.setattr(yt, "_fetch_info", boom)
    monkeypatch.setattr(yt, "BREAKER_THRESHOLD", 2)

    yt.get_stock_quote("AAPL")
    yt.get_stock_quote("MSFT")
    assert len(calls) == 6

    result = yt.get_stock_quote("NVDA")
    assert result.status == "unavailable"
    assert result.error == "provider_unavailable"
    assert len(calls) == 6


def test_unexpected_normaliser_error_is_contained(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_info", lambda symbol: info())
    monkeypatch.setattr(
        yt, "_normalize_quote", lambda payload: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    result = yt.get_stock_quote("AAPL")
    assert result.status == "error"
    assert result.error == "malformed_response"
