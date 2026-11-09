"""Tool package: registry, hardened yfinance tools and the tool dispatch table."""

from __future__ import annotations

from typing import Any, Callable

from app.tools import registry
from app.tools.yfinance_tools import (
    get_company_news,
    get_financial_ratios,
    get_historical_prices,
    get_stock_quote,
)

TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_stock_quote": get_stock_quote,
    "get_financial_ratios": get_financial_ratios,
    "get_historical_prices": get_historical_prices,
    "get_company_news": get_company_news,
}

__all__ = [
    "TOOL_FUNCTIONS",
    "get_company_news",
    "get_financial_ratios",
    "get_historical_prices",
    "get_stock_quote",
    "registry",
]
