"""Record grounded baseline data for one ticker (proto fixture recorder).

Calls every tool once, records the results in the evidence ledger and writes a
JSON artifact. No model is involved: this is the substrate that fixtures, the
gold planner and evaluation replays will build on.

Example:
    python -m scripts.run_baseline --ticker AAPL --period 1y
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evidence.calcs import CalculationError
from app.evidence.calcs import apply as apply_calc
from app.evidence.ledger import EvidenceLedger
from app.evidence.render import render_ledger
from app.schemas.report import DISCLAIMER
from app.tools import TOOL_FUNCTIONS
from app.tools.registry import DEFAULT_PERIOD, evidence_fields


def collect(
    ticker: str, period: str, news_limit: int
) -> tuple[list[dict[str, Any]], EvidenceLedger]:
    """Run the four data tools and record their evidence."""

    calls = [
        ("c1", "get_stock_quote", {"ticker": ticker}),
        ("c2", "get_financial_ratios", {"ticker": ticker}),
        ("c3", "get_historical_prices", {"ticker": ticker, "period": period}),
        ("c4", "get_company_news", {"ticker": ticker, "limit": news_limit}),
    ]
    ledger = EvidenceLedger()
    results: list[dict[str, Any]] = []
    for call_id, tool, args in calls:
        result = TOOL_FUNCTIONS[tool](**args)
        results.append({"call_id": call_id, **result.model_dump(mode="json")})
        ledger.add_from_tool_result(result, call_id, evidence_fields(tool, result.data))
    return results, ledger


def add_derived_example(ledger: EvidenceLedger) -> None:
    """Register the price-vs-52-week-low distance when both values exist."""

    price = next((entry for entry in ledger if entry.label == "price" and not entry.is_null), None)
    low = next(
        (entry for entry in ledger if entry.label == "fifty_two_week_low" and not entry.is_null),
        None,
    )
    if price is None or low is None:
        return
    try:
        apply_calc(ledger, "pct_change", low.id, price.id, label="price_vs_52w_low")
    except CalculationError:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--news-limit", type=int, default=5)
    parser.add_argument("--output", default="artifacts/baseline")
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    results, ledger = collect(ticker, args.period, args.news_limit)
    add_derived_example(ledger)

    report = {
        "schema_version": "2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "period": args.period,
        "tool_results": results,
        "evidence": ledger.to_dicts(),
        "evidence_text": render_ledger(ledger),
        "disclaimer": DISCLAIMER,
    }

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"{ticker.lower()}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved baseline data to {output_path}")
    print(f"Evidence entries: {len(ledger)} ({', '.join(ledger.ids())})")
    for result in results:
        detail = result["error"] or ""
        print(f"  {result['call_id']} {result['tool']}: {result['status']} {detail}".rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
