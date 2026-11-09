"""Record real tool responses into ``data/fixtures`` (plan section 11.3, step A).

Fixtures are a frozen snapshot of what the providers actually returned, failures
included. Recording is resumable: a usable fixture is skipped, an unusable one is
retried, and ``--force`` re-records everything.

``--verify`` proves the acceptance criterion for WS2: every fixture replays
through the ledger deterministically with network access disabled.

Usage:
    python pipelines/01_record_fixtures.py --limit 4
    python pipelines/01_record_fixtures.py
    python pipelines/01_record_fixtures.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python pipelines/01_record_fixtures.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evidence.ledger import EvidenceLedger  # noqa: E402
from app.evidence.render import render_ledger  # noqa: E402
from app.tools import TOOL_FUNCTIONS  # noqa: E402
from app.tools.registry import evidence_fields  # noqa: E402
from app.tools.replay import (  # noqa: E402
    DEFAULT_FIXTURES,
    MANIFEST_NAME,
    FixtureStore,
    fixture_filename,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TICKERS_FILE = PROJECT_ROOT / "data" / "seeds" / "tickers.json"
FIXTURES_DIR = DEFAULT_FIXTURES

TOOLS: tuple[str, ...] = (
    "get_stock_quote",
    "get_financial_ratios",
    "get_historical_prices",
    "get_company_news",
)
DEFAULT_PERIODS: tuple[str, ...] = ("1mo", "1y")
NEWS_LIMIT = 5
#: Recorded results with these statuses count as a finished fixture.
USABLE_STATUSES = frozenset({"ok", "stale"})


def load_tickers(limit: int | None = None) -> list[dict[str, Any]]:
    payload = json.loads(TICKERS_FILE.read_text(encoding="utf-8"))
    tickers = list(payload["tickers"])
    return tickers[:limit] if limit else tickers


def tool_args(tool: str, symbol: str, period: str) -> dict[str, Any]:
    if tool == "get_historical_prices":
        return {"ticker": symbol, "period": period}
    if tool == "get_company_news":
        return {"ticker": symbol, "limit": NEWS_LIMIT}
    return {"ticker": symbol}


def record_one(tool: str, symbol: str, period: str) -> dict[str, Any]:
    args = tool_args(tool, symbol, period)
    result = TOOL_FUNCTIONS[tool](**args)
    return {
        "ticker": symbol,
        "tool": tool,
        "period": period if tool == "get_historical_prices" else None,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "result": result.model_dump(mode="json"),
    }


def is_usable(payload: dict[str, Any]) -> bool:
    status = payload.get("result", {}).get("status")
    return status in USABLE_STATUSES


def write_fixture(payload: dict[str, Any]) -> Path:
    path = FIXTURES_DIR / fixture_filename(
        payload["tool"], payload["ticker"], payload.get("period")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_manifest() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        if path.name == MANIFEST_NAME:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["result"]
        entries.append(
            {
                "ticker": payload["ticker"],
                "tool": payload["tool"],
                "period": payload.get("period"),
                "file": path.name,
                "status": result.get("status"),
                "error": result.get("error"),
                "as_of": result.get("as_of"),
                "retrieved_at": result.get("retrieved_at"),
            }
        )
        digest.update(path.name.encode("utf-8"))
        digest.update(json.dumps(result, sort_keys=True).encode("utf-8"))

    periods = sorted({entry["period"] for entry in entries if entry["period"]})
    return {
        "snapshot_date": datetime.now(timezone.utc).date().isoformat(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "provider": "yfinance",
        "periods": periods,
        "fixture_count": len(entries),
        "tickers": sorted({entry["ticker"] for entry in entries}),
        "content_hash": digest.hexdigest(),
        "entries": entries,
    }


def write_manifest() -> dict[str, Any]:
    manifest = build_manifest()
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def record(
    *,
    force: bool,
    limit: int | None,
    symbols: list[str] | None,
    periods: tuple[str, ...],
) -> int:
    tickers = load_tickers(limit)
    if symbols:
        tickers = [entry for entry in tickers if entry["symbol"] in symbols]
    total = len(tickers)
    recorded = skipped = failed = 0

    for position, entry in enumerate(tickers, start=1):
        symbol = entry["symbol"]
        statuses: list[str] = []
        for tool in TOOLS:
            for period in periods if tool == "get_historical_prices" else ("",):
                path = FIXTURES_DIR / fixture_filename(tool, symbol, period or None)
                if path.exists() and not force:
                    try:
                        if is_usable(json.loads(path.read_text(encoding="utf-8"))):
                            skipped += 1
                            statuses.append(f"{tool}:skip")
                            continue
                    except (json.JSONDecodeError, KeyError):
                        pass
                payload = record_one(tool, symbol, period)
                write_fixture(payload)
                recorded += 1
                status = payload["result"]["status"]
                statuses.append(f"{tool}:{status}")
                if status == "error":
                    failed += 1
        write_manifest()
        print(f"[{position}/{total}] {symbol} " + " ".join(statuses), flush=True)

    manifest = write_manifest()
    print(
        f"recorded={recorded} skipped={skipped} errors={failed} "
        f"fixtures={manifest['fixture_count']} hash={manifest['content_hash'][:12]}"
    )
    print("RECORDING COMPLETE")
    return 0


def _disable_network() -> None:
    def blocked(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("network access is disabled during fixture verification")

    socket.socket = blocked  # type: ignore[assignment]
    socket.create_connection = blocked  # type: ignore[assignment]


def _replay_digest(store: FixtureStore) -> str:
    hasher = hashlib.sha256()
    for entry in store.entries():
        result = store.result(entry["tool"], entry["ticker"], entry.get("period"))
        ledger = EvidenceLedger()
        ledger.add_from_tool_result(
            result, "replay", evidence_fields(entry["tool"], result.data)
        )
        hasher.update(render_ledger(ledger).encode("utf-8"))
    return hasher.hexdigest()


def verify() -> int:
    _disable_network()
    store = FixtureStore(FIXTURES_DIR)
    if store.count() == 0:
        print("no fixtures recorded yet; run without --verify first")
        return 1
    first = _replay_digest(store)
    second = _replay_digest(store)
    deterministic = first == second
    complete = store.count() == len(
        [path for path in FIXTURES_DIR.glob("*.json") if path.name != MANIFEST_NAME]
    )
    print(f"fixtures={store.count()} tickers={len(store.tickers())} periods={store.periods()}")
    print(f"replay_digest={first}")
    print(f"deterministic={'PASS' if deterministic else 'FAIL'}")
    print(f"manifest_complete={'PASS' if complete else 'FAIL'}")
    return 0 if deterministic and complete else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-record usable fixtures")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--symbols", default=None, help="comma-separated subset")
    parser.add_argument("--periods", default=",".join(DEFAULT_PERIODS))
    parser.add_argument("--verify", action="store_true", help="replay offline and check")
    args = parser.parse_args()

    if args.verify:
        return verify()

    symbols = [item.strip().upper() for item in args.symbols.split(",")] if args.symbols else None
    periods = tuple(item.strip() for item in args.periods.split(",") if item.strip())
    return record(force=args.force, limit=args.limit, symbols=symbols, periods=periods)


if __name__ == "__main__":
    raise SystemExit(main())
