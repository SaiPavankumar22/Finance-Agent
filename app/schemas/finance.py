"""Versioned schemas for finance-tool outputs and grounded reports."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    retrieved_at: datetime
    source_url: Optional[str] = None
    source_timestamp: Optional[datetime] = None


class StockQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    ticker: str
    currency: Optional[str] = None
    price: Optional[float] = Field(default=None, ge=0)
    volume: Optional[int] = Field(default=None, ge=0)
    market_cap: Optional[float] = Field(default=None, ge=0)
    fifty_two_week_low: Optional[float] = Field(default=None, ge=0)
    fifty_two_week_high: Optional[float] = Field(default=None, ge=0)
    source: SourceRef


class FinancialRatios(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    ticker: str
    pe_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    enterprise_value_to_ebitda: Optional[float] = None
    debt_to_equity: Optional[float] = None
    return_on_equity: Optional[float] = None
    source: SourceRef


class PricePoint(BaseModel):
    date: str
    close: float = Field(ge=0)


class HistoricalPrices(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    ticker: str
    period: str
    prices: list[PricePoint]
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_20: Optional[float] = None
    annualized_volatility: Optional[float] = Field(default=None, ge=0)
    source: SourceRef


class NewsItem(BaseModel):
    title: str
    publisher: Optional[str] = None
    link: Optional[str] = None
    published_at: Optional[datetime] = None


class CompanyNews(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    ticker: str
    items: list[NewsItem]
    unavailable_reason: Optional[str] = None
    source: SourceRef


class GroundedReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    ticker: str
    summary: str
    facts: list[str]
    calculations: list[str]
    limitations: list[str]
    evidence_refs: list[str]
    disclaimer: str = (
        "This is informational research, not personalized investment advice."
    )
