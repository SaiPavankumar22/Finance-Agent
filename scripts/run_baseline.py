"""Collect a grounded finance baseline report from yfinance.

Example:
    python -m scripts.run_baseline --ticker AAPL --period 1y
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from app.evidence.validator import EvidenceLedger, validate_numeric_claims
from app.tools.yfinance_tools import (
    get_company_news,
    get_financial_ratios,
    get_historical_prices,
    get_stock_quote,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--news-limit", type=int, default=5)
    parser.add_argument("--output", default="artifacts/baseline")
    args = parser.parse_args()

    quote = get_stock_quote(args.ticker)
    ratios = get_financial_ratios(args.ticker)
    history = get_historical_prices(args.ticker, args.period)
    news = get_company_news(args.ticker, args.news_limit)

    ledger = EvidenceLedger()
    ledger.add("quote", quote.model_dump(mode="json"))
    ledger.add("ratios", ratios.model_dump(mode="json"))
    ledger.add("history", history.model_dump(mode="json"))
    ledger.add("news", news.model_dump(mode="json"))

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker": quote.ticker,
        "quote": quote.model_dump(mode="json"),
        "ratios": ratios.model_dump(mode="json"),
        "history": history.model_dump(mode="json"),
        "news": news.model_dump(mode="json"),
        "evidence_refs": ledger.references(),
        "disclaimer": "Informational research only; not personalized investment advice.",
    }
    validate_numeric_claims(report, ledger)

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"{quote.ticker.lower()}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved grounded baseline to {output_path}")
    print(f"Evidence sources: {', '.join(ledger.references())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
