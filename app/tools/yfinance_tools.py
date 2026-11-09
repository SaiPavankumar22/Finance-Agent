"""Validated read-only finance tools backed by yfinance."""

from datetime import datetime, timezone
import math
import re
from typing import Any

import pandas as pd
import yfinance as yf

from app.schemas.finance import (
    CompanyNews,
    FinancialRatios,
    HistoricalPrices,
    NewsItem,
    PricePoint,
    SourceRef,
    StockQuote,
)


TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
ALLOWED_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"}


def _ticker(value: str) -> str:
    ticker = value.strip().upper()
    if not TICKER_PATTERN.fullmatch(ticker):
        raise ValueError("ticker must be 1-10 characters such as AAPL or BRK-B")
    return ticker


def _source() -> SourceRef:
    return SourceRef(
        provider="Yahoo Finance via yfinance",
        retrieved_at=datetime.now(timezone.utc),
        source_url="https://finance.yahoo.com/",
    )


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def get_stock_quote(ticker: str) -> StockQuote:
    symbol = _ticker(ticker)
    security = yf.Ticker(symbol)
    info = security.info or {}
    fast_info = security.fast_info
    price = _number(info.get("regularMarketPrice"))
    if price is None:
        price = _number(fast_info.get("last_price"))

    return StockQuote(
        ticker=symbol,
        currency=info.get("currency"),
        price=price,
        volume=_int(info.get("regularMarketVolume") or info.get("volume")),
        market_cap=_number(info.get("marketCap")),
        fifty_two_week_low=_number(info.get("fiftyTwoWeekLow")),
        fifty_two_week_high=_number(info.get("fiftyTwoWeekHigh")),
        source=_source(),
    )


def get_financial_ratios(ticker: str) -> FinancialRatios:
    symbol = _ticker(ticker)
    info = yf.Ticker(symbol).info or {}
    return FinancialRatios(
        ticker=symbol,
        pe_ratio=_number(info.get("trailingPE") or info.get("forwardPE")),
        price_to_book=_number(info.get("priceToBook")),
        enterprise_value_to_ebitda=_number(info.get("enterpriseToEbitda")),
        debt_to_equity=_number(info.get("debtToEquity")),
        return_on_equity=_number(info.get("returnOnEquity")),
        source=_source(),
    )


def get_historical_prices(ticker: str, period: str = "1y") -> HistoricalPrices:
    symbol = _ticker(ticker)
    if period not in ALLOWED_PERIODS:
        raise ValueError(f"period must be one of: {', '.join(sorted(ALLOWED_PERIODS))}")

    history = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if history.empty or "Close" not in history:
        raise ValueError(f"no historical prices returned for {symbol}")

    closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
    returns = closes.pct_change().dropna()
    return HistoricalPrices(
        ticker=symbol,
        period=period,
        prices=[
            PricePoint(date=index.date().isoformat(), close=float(value))
            for index, value in closes.items()
        ],
        sma_20=_number(closes.rolling(20).mean().iloc[-1]),
        sma_50=_number(closes.rolling(50).mean().iloc[-1]),
        ema_20=_number(closes.ewm(span=20, adjust=False).mean().iloc[-1]),
        annualized_volatility=_number(returns.std() * (252**0.5)),
        source=_source(),
    )


def get_company_news(ticker: str, limit: int = 5) -> CompanyNews:
    symbol = _ticker(ticker)
    if not 1 <= limit <= 25:
        raise ValueError("limit must be between 1 and 25")

    try:
        raw_items = yf.Ticker(symbol).news or []
        unavailable_reason = None
    except Exception as error:
        raw_items = []
        unavailable_reason = f"{type(error).__name__}: {error}"
    items: list[NewsItem] = []
    for raw in raw_items[:limit]:
        content = raw.get("content", raw)
        title = content.get("title") or raw.get("title")
        if not title:
            continue
        published = content.get("pubDate") or content.get("providerPublishTime")
        published_at = None
        if isinstance(published, (int, float)):
            published_at = datetime.fromtimestamp(published, tz=timezone.utc)
        elif isinstance(published, str):
            try:
                published_at = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except ValueError:
                published_at = None
        provider = content.get("provider") or raw.get("publisher")
        publisher = provider.get("displayName") if isinstance(provider, dict) else provider
        link = content.get("canonicalUrl") or content.get("clickThroughUrl")
        if isinstance(link, dict):
            link = link.get("url")
        items.append(
            NewsItem(
                title=str(title),
                publisher=publisher,
                link=link,
                published_at=published_at,
            )
        )

    return CompanyNews(
        ticker=symbol,
        items=items,
        unavailable_reason=unavailable_reason,
        source=_source(),
    )
