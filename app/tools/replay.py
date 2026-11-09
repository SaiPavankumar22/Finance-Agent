"""Offline fixture replay: evaluation never touches the network (plan section 9).

yfinance is unofficial and breaks; recorded fixtures are the frozen substrate for
the test set, the gold planner and every eval replay. This module owns the
on-disk naming convention and returns recorded ``ToolResult`` objects, or an
explicit ``fixture_missing`` result - it never falls back to a live call, because
a silent network call would make an "offline" evaluation not offline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas.tools import ToolResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES = PROJECT_ROOT / "data" / "fixtures"
MANIFEST_NAME = "manifest.json"
#: Fixed timestamp used for synthesised results, so replay stays deterministic.
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fixture_filename(tool: str, ticker: str, period: str | None = None) -> str:
    """``AAPL_get_historical_prices_1y.json`` style file name."""

    suffix = f"_{period}" if period else ""
    return f"{ticker}_{tool}{suffix}.json"


class FixtureStore:
    """Read access to a recorded fixture directory."""

    def __init__(self, root: Path | str = DEFAULT_FIXTURES) -> None:
        self.root = Path(root)

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"fixture_count": 0, "tickers": [], "entries": [], "periods": []}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def entries(self) -> list[dict[str, Any]]:
        return list(self.manifest().get("entries", []))

    def count(self) -> int:
        return len(self.entries())

    def tickers(self) -> list[str]:
        return list(self.manifest().get("tickers", []))

    def periods(self) -> list[str]:
        return list(self.manifest().get("periods", []))

    def path(self, tool: str, ticker: str, period: str | None = None) -> Path:
        return self.root / fixture_filename(tool, ticker, period)

    def has(self, tool: str, ticker: str, period: str | None = None) -> bool:
        return self.path(tool, ticker, period).exists()

    def result(self, tool: str, ticker: str, period: str | None = None) -> ToolResult:
        """Recorded result for one call, or an explicit ``fixture_missing``."""

        path = self.path(tool, ticker, period)
        args: dict[str, Any] = {"ticker": ticker}
        if period:
            args["period"] = period
        if not path.exists():
            return ToolResult(
                status="unavailable",
                tool=tool,
                args=args,
                data=None,
                source="fixture",
                retrieved_at=EPOCH,
                as_of=None,
                error="fixture_missing",
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ToolResult.model_validate(payload["result"])
