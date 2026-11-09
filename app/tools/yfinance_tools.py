"""Validated read-only finance tools backed by yfinance (plan section 9).

Contract: **no tool raises**. Every call resolves to a ``ToolResult`` with a
machine-readable status and error code. Operational rules:

* 8s timeout per attempt, 2 retries with exponential backoff.
* TTL cache: quotes 60s, ratios 10m, history 24h, news 15m. Cached results keep
  their original ``retrieved_at``, because a cached value is not a new fetch.
* Staleness: an ``as_of`` older than the threshold downgrades the result to
  ``status="stale"`` instead of pretending it is fresh.
* Circuit breaker: after ``BREAKER_THRESHOLD`` consecutive provider outages in a
  window, calls return ``unavailable`` immediately until the cooldown expires.
* Empty or unrecognisable payloads become ``error`` / ``invalid_ticker``.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

import pandas as pd
import yfinance as yf

from app.schemas.tools import ToolResult
from app.tools.registry import ALLOWED_PERIODS, DEFAULT_PERIOD, MAX_NEWS_LIMIT

logger = logging.getLogger(__name__)

SOURCE = "yfinance"
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

CALL_TIMEOUT_SECONDS = 8.0
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0

BREAKER_THRESHOLD = 5
BREAKER_COOLDOWN_SECONDS = 300.0

CACHE_TTL_SECONDS: dict[str, float] = {
    "get_stock_quote": 60.0,
    "get_financial_ratios": 600.0,
    "get_historical_prices": 86_400.0,
    "get_company_news": 900.0,
}

STALENESS_THRESHOLDS: dict[str, timedelta] = {
    "get_stock_quote": timedelta(days=5),
    "get_financial_ratios": timedelta(days=120),
    "get_historical_prices": timedelta(days=10),
    "get_company_news": timedelta(days=90),
}

#: Error codes that mean "this request was wrong" (status ``error``) rather than
#: "the provider is down" (status ``unavailable``).
CLIENT_ERROR_CODES = frozenset(
    {"invalid_ticker", "invalid_period", "invalid_limit", "malformed_response"}
)
#: Codes that count toward the circuit breaker.
OUTAGE_CODES = frozenset({"provider_timeout", "provider_unavailable"})


class ProviderFailure(Exception):
    """A provider call failed in a classified way."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


_CACHE: dict[tuple[str, tuple[tuple[str, str], ...]], tuple[float, ToolResult]] = {}
_CACHE_LOCK = threading.Lock()

_BREAKER_LOCK = threading.Lock()
#: Breaker state is per tool: a dead news endpoint must not blind every other tool.
_BREAKER_FAILURES: dict[str, int] = {}
_BREAKER_OPEN_UNTIL: dict[str, float] = {}

#: One ``.info`` fetch serves both the quote and the ratio tools.
INFO_TTL_SECONDS = 60.0
_INFO_CACHE: dict[str, tuple[float, Mapping[str, Any]]] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def reset_state() -> None:
    """Clear caches and breakers (used by tests and eval replays)."""

    with _CACHE_LOCK:
        _CACHE.clear()
    with _BREAKER_LOCK:
        _BREAKER_FAILURES.clear()
        _BREAKER_OPEN_UNTIL.clear()
    with _INFO_LOCK:
        _INFO_CACHE.clear()


# --------------------------------------------------------------------------- #
# Guards: timeout, retry, cache, breaker
# --------------------------------------------------------------------------- #
def _guarded(call: Callable[[], Any], timeout: float) -> Any:
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(call).result(timeout=timeout)
    finally:
        pool.shutdown(wait=False)


def _with_retries(
    call: Callable[[], Any],
    *,
    timeout: float | None = None,
    retries: int | None = None,
) -> Any:
    timeout = CALL_TIMEOUT_SECONDS if timeout is None else timeout
    retries = MAX_RETRIES if retries is None else retries
    last: ProviderFailure | None = None
    for attempt in range(retries + 1):
        try:
            return _guarded(call, timeout)
        except (FutureTimeout, TimeoutError):
            last = ProviderFailure("provider_timeout", f"no response within {timeout}s")
        except Exception as error:  # provider libraries raise many unrelated types
            last = ProviderFailure("provider_unavailable", f"{type(error).__name__}: {error}")
        if attempt < retries:
            time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
    assert last is not None
    raise last


def _cache_key(tool: str, args: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    return (tool, tuple(sorted((name, str(value)) for name, value in args.items())))


def _cache_get(key: tuple[str, tuple[tuple[str, str], ...]], now: datetime) -> ToolResult | None:
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        expires_at, result = entry
        if expires_at <= now.timestamp():
            _CACHE.pop(key, None)
            return None
        return result


def _cache_set(
    key: tuple[str, tuple[tuple[str, str], ...]],
    result: ToolResult,
    now: datetime,
    ttl: float,
) -> None:
    if ttl <= 0:
        return
    with _CACHE_LOCK:
        _CACHE[key] = (now.timestamp() + ttl, result)


def _breaker_open(tool: str, now: datetime) -> bool:
    with _BREAKER_LOCK:
        return now.timestamp() < _BREAKER_OPEN_UNTIL.get(tool, 0.0)


def _record_failure(tool: str) -> None:
    with _BREAKER_LOCK:
        failures = _BREAKER_FAILURES.get(tool, 0) + 1
        _BREAKER_FAILURES[tool] = failures
        if failures >= BREAKER_THRESHOLD:
            _BREAKER_OPEN_UNTIL[tool] = time.time() + BREAKER_COOLDOWN_SECONDS


def _record_success(tool: str) -> None:
    with _BREAKER_LOCK:
        _BREAKER_FAILURES.pop(tool, None)
        _BREAKER_OPEN_UNTIL.pop(tool, None)


# --------------------------------------------------------------------------- #
# Result construction
# --------------------------------------------------------------------------- #
def _failure(tool: str, args: dict[str, Any], code: str, detail: str, now: datetime) -> ToolResult:
    status = "error" if code in CLIENT_ERROR_CODES else "unavailable"
    logger.warning("%s failed: %s (%s)", tool, code, detail)
    return ToolResult(
        status=status,
        tool=tool,
        args=dict(args),
        data=None,
        source=SOURCE,
        retrieved_at=now,
        as_of=None,
        error=code,
    )


def _is_stale(tool: str, as_of: datetime | None, now: datetime) -> bool:
    if as_of is None:
        return False
    threshold = STALENESS_THRESHOLDS.get(tool)
    return threshold is not None and now - as_of > threshold


def _run(
    tool: str,
    raw_args: dict[str, Any],
    fetch: Callable[[], Any],
    normalize: Callable[[Any], tuple[dict[str, Any], datetime | None]],
) -> ToolResult:
    args = dict(raw_args)
    now = _now()
    if _breaker_open(tool, now):
        return _failure(tool, args, "provider_unavailable", "circuit breaker open", now)

    key = _cache_key(tool, args)
    cached = _cache_get(key, now)
    if cached is not None:
        return cached

    try:
        data, as_of = normalize(_with_retries(fetch))
    except ProviderFailure as failure:
        if failure.code in OUTAGE_CODES:
            _record_failure(tool)
        return _failure(tool, args, failure.code, failure.detail, now)
    except Exception as error:  # normalisation must never escape as a crash
        _record_failure(tool)
        return _failure(tool, args, "malformed_response", f"{type(error).__name__}: {error}", now)

    _record_success(tool)
    result = ToolResult(
        status="stale" if _is_stale(tool, as_of, now) else "ok",
        tool=tool,
        args=args,
        data=data,
        source=SOURCE,
        retrieved_at=now,
        as_of=as_of,
        error=None,
    )
    _cache_set(key, result, now, CACHE_TTL_SECONDS.get(tool, 0.0))
    return result


# --------------------------------------------------------------------------- #
# Payload helpers
# --------------------------------------------------------------------------- #
def _mapping(payload: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ProviderFailure("malformed_response", f"{label} payload is not a mapping")
    return payload


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_number(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        number = _number(payload.get(key))
        if number is not None:
            return number
    return None


def _first_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    number = _first_number(payload, *keys)
    return int(number) if number is not None else None


def _epoch(value: Any) -> datetime | None:
    number = _number(value)
    if number is None or number <= 0:
        return None
    return datetime.fromtimestamp(number, tz=timezone.utc)


def _looks_empty(payload: Mapping[str, Any]) -> bool:
    recognizable = {
        "symbol",
        "shortName",
        "longName",
        "currency",
        "regularMarketPrice",
        "currentPrice",
        "marketCap",
    }
    return not any(key in payload for key in recognizable)


# --------------------------------------------------------------------------- #
# Normalisers (pure: exercised by tests without network access)
# --------------------------------------------------------------------------- #
def _normalize_quote(payload: Any) -> tuple[dict[str, Any], datetime | None]:
    info = _mapping(payload, "quote")
    price = _first_number(info, "regularMarketPrice", "currentPrice")
    if price is None and _looks_empty(info):
        raise ProviderFailure("invalid_ticker", "no quote fields returned")
    data = {
        "ticker": info.get("symbol"),
        "currency": info.get("currency"),
        "price": price,
        "volume": _first_int(info, "regularMarketVolume", "volume"),
        "market_cap": _number(info.get("marketCap")),
        "fifty_two_week_low": _number(info.get("fiftyTwoWeekLow")),
        "fifty_two_week_high": _number(info.get("fiftyTwoWeekHigh")),
    }
    as_of = _epoch(info.get("regularMarketTime")) or _epoch(info.get("regularMarketTradeTime"))
    return data, as_of


def _normalize_ratios(payload: Any) -> tuple[dict[str, Any], datetime | None]:
    info = _mapping(payload, "ratios")
    if _looks_empty(info):
        raise ProviderFailure("invalid_ticker", "no ratio fields returned")
    data = {
        "ticker": info.get("symbol"),
        "currency": info.get("currency"),
        "pe_ratio": _first_number(info, "trailingPE", "forwardPE"),
        "price_to_book": _number(info.get("priceToBook")),
        "enterprise_value_to_ebitda": _number(info.get("enterpriseToEbitda")),
        "debt_to_equity": _number(info.get("debtToEquity")),
        "return_on_equity": _number(info.get("returnOnEquity")),
    }
    # Yahoo exposes no timestamp for these aggregates; the retrieval time is the
    # only honest freshness signal, so as_of stays null.
    return data, None


def _normalize_history(payload: Any, period: str) -> tuple[dict[str, Any], datetime | None]:
    if payload is None or not hasattr(payload, "empty"):
        raise ProviderFailure("malformed_response", "history payload is not a frame")
    frame = payload
    if frame.empty:
        raise ProviderFailure("invalid_ticker", "no historical rows returned")
    if "Close" not in frame.columns:
        raise ProviderFailure("malformed_response", "history payload has no Close column")

    closes = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if closes.empty:
        raise ProviderFailure("invalid_ticker", "no close prices returned")

    returns = closes.pct_change().dropna()
    last_index = closes.index[-1]
    last_date = last_index.date().isoformat() if hasattr(last_index, "date") else str(last_index)
    data = {
        "ticker": None,
        "period": period,
        "prices": [
            {
                "date": index.date().isoformat() if hasattr(index, "date") else str(index),
                "close": float(value),
            }
            for index, value in closes.items()
        ],
        "last_close": float(closes.iloc[-1]),
        "last_date": last_date,
        "observations": int(len(closes)),
        "sma_20": _number(closes.rolling(20).mean().iloc[-1]) if len(closes) >= 20 else None,
        "sma_50": _number(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else None,
        "ema_20": _number(closes.ewm(span=20, adjust=False).mean().iloc[-1])
        if len(closes) >= 20
        else None,
        "annualized_volatility": _number(returns.std() * (252**0.5)),
        "currency": None,
    }
    as_of = (
        datetime.combine(last_index.date(), datetime.min.time(), tzinfo=timezone.utc)
        if hasattr(last_index, "date")
        else None
    )
    return data, as_of


def _normalize_news(payload: Any, limit: int) -> tuple[dict[str, Any], datetime | None]:
    if payload is None:
        raise ProviderFailure("provider_unavailable", "news payload missing")
    if not isinstance(payload, (list, tuple)):
        raise ProviderFailure("malformed_response", "news payload is not a list")

    items: list[dict[str, Any]] = []
    for raw in list(payload)[:limit]:
        if not isinstance(raw, Mapping):
            continue
        content = raw.get("content") if isinstance(raw.get("content"), Mapping) else raw
        title = content.get("title") or raw.get("title")
        if not title:
            continue
        published = content.get("pubDate") or content.get("providerPublishTime")
        published_at: datetime | None = None
        if isinstance(published, (int, float)) and not isinstance(published, bool):
            published_at = datetime.fromtimestamp(published, tz=timezone.utc)
        elif isinstance(published, str):
            try:
                published_at = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except ValueError:
                published_at = None
        provider = content.get("provider") or raw.get("publisher")
        publisher = provider.get("displayName") if isinstance(provider, Mapping) else provider
        link = content.get("canonicalUrl") or content.get("clickThroughUrl")
        if isinstance(link, Mapping):
            link = link.get("url")
        items.append(
            {
                "title": str(title),
                "publisher": publisher,
                "url": link,
                "published_at": published_at.isoformat() if published_at else None,
            }
        )

    if not items:
        raise ProviderFailure("no_news_found", "provider returned no usable headlines")

    timestamps = [item["published_at"] for item in items if item["published_at"]]
    as_of = datetime.fromisoformat(max(timestamps)) if timestamps else None
    return {"ticker": None, "count": len(items), "items": items}, as_of


# --------------------------------------------------------------------------- #
# Fetchers (the only code that touches the network)
# --------------------------------------------------------------------------- #
def _fetch_info(symbol: str) -> Any:
    return yf.Ticker(symbol).info


_INFO_LOCK = threading.Lock()


def _fetch_info_cached(symbol: str) -> Any:
    """One shared ``.info`` payload per symbol, so quote + ratios cost one call."""

    now = time.time()
    with _INFO_LOCK:
        hit = _INFO_CACHE.get(symbol)
        if hit is not None and hit[0] > now:
            return hit[1]
    payload = _fetch_info(symbol)
    if isinstance(payload, Mapping):
        with _INFO_LOCK:
            _INFO_CACHE[symbol] = (now + INFO_TTL_SECONDS, payload)
    return payload


def _fetch_history(symbol: str, period: str) -> Any:
    return yf.Ticker(symbol).history(period=period, auto_adjust=True)


def _fetch_news(symbol: str) -> Any:
    return yf.Ticker(symbol).news or []


# --------------------------------------------------------------------------- #
# Public tools
# --------------------------------------------------------------------------- #
def _validate_ticker(tool: str, raw: Any) -> tuple[str | None, ToolResult | None]:
    if not isinstance(raw, str) or not raw.strip():
        return None, _failure(
            tool, {"ticker": raw}, "invalid_ticker", "ticker is not a string", _now()
        )
    symbol = raw.strip().upper()
    if not TICKER_PATTERN.fullmatch(symbol):
        return None, _failure(
            tool, {"ticker": symbol}, "invalid_ticker", "ticker fails symbol pattern", _now()
        )
    return symbol, None


def get_stock_quote(ticker: str) -> ToolResult:
    """Latest quote: price, currency, volume, market cap, 52-week range."""

    symbol, failure = _validate_ticker("get_stock_quote", ticker)
    if failure is not None:
        return failure
    assert symbol is not None
    return _run(
        "get_stock_quote", {"ticker": symbol}, lambda: _fetch_info_cached(symbol), _normalize_quote
    )


def get_financial_ratios(ticker: str) -> ToolResult:
    """Valuation and leverage ratios; missing fields stay null, never zero."""

    symbol, failure = _validate_ticker("get_financial_ratios", ticker)
    if failure is not None:
        return failure
    assert symbol is not None
    return _run(
        "get_financial_ratios",
        {"ticker": symbol},
        lambda: _fetch_info_cached(symbol),
        _normalize_ratios,
    )


def get_historical_prices(ticker: str, period: str = DEFAULT_PERIOD) -> ToolResult:
    """Daily closes plus SMA/EMA and annualized volatility for one period."""

    symbol, failure = _validate_ticker("get_historical_prices", ticker)
    if failure is not None:
        return failure
    args = {"ticker": symbol, "period": period}
    if period not in ALLOWED_PERIODS:
        return _failure(
            "get_historical_prices",
            args,
            "invalid_period",
            f"period must be one of {ALLOWED_PERIODS}",
            _now(),
        )
    assert symbol is not None
    return _run(
        "get_historical_prices",
        args,
        lambda: _fetch_history(symbol, period),
        lambda payload: _normalize_history(payload, period),
    )


def get_company_news(ticker: str, limit: int = 5) -> ToolResult:
    """Recent headlines, or ``unavailable`` when the provider returns none."""

    symbol, failure = _validate_ticker("get_company_news", ticker)
    if failure is not None:
        return failure
    args = {"ticker": symbol, "limit": limit}
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_NEWS_LIMIT:
        return _failure(
            "get_company_news", args, "invalid_limit", f"limit must be 1..{MAX_NEWS_LIMIT}", _now()
        )
    assert symbol is not None
    return _run(
        "get_company_news",
        args,
        lambda: _fetch_news(symbol),
        lambda payload: _normalize_news(payload, limit),
    )
