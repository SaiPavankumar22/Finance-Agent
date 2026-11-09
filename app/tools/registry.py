"""Tool registry: the small, fixed tool surface the model may call (plan section 10).

A 3B model degrades quickly with long tool lists, so this stays at four data
tools plus the terminal ``finish`` action. The registry also owns which payload
fields become evidence and how they are labelled and formatted, which keeps that
domain knowledge out of the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.evidence.ledger import FieldSpec, dig

FINISH_TOOL = "finish"
ALLOWED_PERIODS: tuple[str, ...] = ("1mo", "3mo", "6mo", "1y", "2y", "5y")
DEFAULT_PERIOD = "1y"
MAX_NEWS_LIMIT = 25
#: How many headlines become citable evidence for one news result.
MAX_CITABLE_HEADLINES = 5


@dataclass(frozen=True, slots=True)
class ArgSpec:
    """One declared argument, used to generate the planning prompt."""

    name: str
    type: str
    required: bool = True
    description: str = ""
    values: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A callable tool and the evidence it produces."""

    name: str
    description: str
    args: tuple[ArgSpec, ...] = ()
    evidence: tuple[FieldSpec, ...] = ()
    terminal: bool = False


TOOL_SPECS: dict[str, ToolSpec] = {
    "get_stock_quote": ToolSpec(
        name="get_stock_quote",
        description="Latest price, currency, volume, market cap and 52-week range.",
        args=(ArgSpec("ticker", "string", description="exchange symbol, e.g. AAPL or SAP.DE"),),
        evidence=(
            FieldSpec("price", style="plain"),
            FieldSpec("market_cap", style="compact"),
            FieldSpec("volume", unit="shares", style="compact", currency_path=None),
            FieldSpec("fifty_two_week_low", style="plain"),
            FieldSpec("fifty_two_week_high", style="plain"),
        ),
    ),
    "get_financial_ratios": ToolSpec(
        name="get_financial_ratios",
        description="P/E, P/B, EV/EBITDA, debt-to-equity and ROE; null when unavailable.",
        args=(ArgSpec("ticker", "string"),),
        evidence=(
            FieldSpec("pe_ratio", unit="ratio", currency_path=None),
            FieldSpec("price_to_book", unit="ratio", currency_path=None),
            FieldSpec("enterprise_value_to_ebitda", unit="ratio", currency_path=None),
            FieldSpec("debt_to_equity", unit="ratio", currency_path=None),
            FieldSpec("return_on_equity", unit="ratio", currency_path=None),
        ),
    ),
    "get_historical_prices": ToolSpec(
        name="get_historical_prices",
        description="Daily closes with SMA/EMA and annualized volatility for a period.",
        args=(
            ArgSpec("ticker", "string"),
            ArgSpec(
                "period",
                "string",
                required=False,
                description="default 1y",
                values=ALLOWED_PERIODS,
            ),
        ),
        evidence=(
            FieldSpec("last_close", style="plain"),
            FieldSpec("sma_20", style="plain"),
            FieldSpec("sma_50", style="plain"),
            FieldSpec("ema_20", style="plain"),
            FieldSpec("annualized_volatility", unit="ratio", currency_path=None),
            FieldSpec("observations", unit="count", currency_path=None),
        ),
    ),
    "get_company_news": ToolSpec(
        name="get_company_news",
        description="Recent headlines with publisher, URL and publication time.",
        args=(
            ArgSpec("ticker", "string"),
            ArgSpec("limit", "integer", required=False, description="default 5, max 25"),
        ),
        evidence=(FieldSpec("count", label="headline_count", unit="count", currency_path=None),),
    ),
    FINISH_TOOL: ToolSpec(
        name=FINISH_TOOL,
        description="Stop calling tools; planning is complete.",
        terminal=True,
    ),
}


def is_known_tool(name: str) -> bool:
    return name in TOOL_SPECS


def evidence_fields(
    tool: str, data: dict[str, Any] | None = None
) -> tuple[FieldSpec, ...]:
    """Evidence specs for a tool, including per-result dynamic fields.

    News headlines are expanded per item so prose can cite a specific headline
    instead of paraphrasing it (which would smuggle ungrounded text into a claim).
    """

    spec = TOOL_SPECS.get(tool)
    if spec is None:
        return ()
    fields = list(spec.evidence)
    if tool == "get_company_news" and data is not None:
        items = dig(data, "items")
        if isinstance(items, (list, tuple)):
            for index in range(min(len(items), MAX_CITABLE_HEADLINES)):
                fields.append(
                    FieldSpec(
                        f"items.{index}.title",
                        label=f"headline_{index + 1}",
                        style="quote",
                        currency_path=None,
                    )
                )
    return tuple(fields)


def tool_prompt() -> str:
    """Compact tool block shared by training and inference prompts."""

    lines = ["Tools:"]
    for spec in TOOL_SPECS.values():
        arguments = ", ".join(
            f"{argument.name}: {argument.type}"
            + (
                " one of " + "|".join(argument.values)
                if argument.values
                else ""
            )
            + ("" if argument.required else " (optional)")
            for argument in spec.args
        )
        lines.append(f"- {spec.name}({arguments or 'no arguments'}): {spec.description}")
    lines.append('Emit exactly one JSON object per turn: {"tool": "<name>", "args": {...}}')
    return "\n".join(lines)
