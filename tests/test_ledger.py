"""Ledger IDs must be stable, and missing values must be recorded, never zeroed."""

from datetime import datetime, timezone

import pytest

from app.evidence.ledger import (
    EvidenceLedger,
    EvidenceValueError,
    FieldSpec,
    UnknownEvidenceError,
    dig,
)
from app.schemas.tools import ToolResult

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


def result(**overrides):
    payload = {
        "status": "ok",
        "tool": "get_stock_quote",
        "args": {"ticker": "AAPL"},
        "data": {"price": 189.23, "currency": "USD", "market_cap": 2.9e12},
        "source": "yfinance",
        "retrieved_at": NOW,
        "as_of": NOW,
    }
    payload.update(overrides)
    return ToolResult(**payload)


def test_ids_are_sequential_and_stable():
    ledger = EvidenceLedger()
    first = ledger.add(label="price", value=1.0)
    second = ledger.add(label="volume", value=2)
    assert (first.id, second.id) == ("E1", "E2")
    assert ledger.get("E1").value == 1.0
    assert ledger.ids() == ["E1", "E2"]


def test_unknown_id_raises():
    ledger = EvidenceLedger()
    with pytest.raises(UnknownEvidenceError):
        ledger.get("E99")
    assert "E99" not in ledger


def test_null_values_are_recorded_as_null():
    ledger = EvidenceLedger()
    entry = ledger.add(label="pe_ratio", value=None, unit="ratio")
    assert entry.is_null is True
    assert entry.value is None
    assert ledger.null_entries() == [entry]


def test_entries_require_a_label():
    with pytest.raises(EvidenceValueError):
        EvidenceLedger().add(label="", value=1.0)


def test_unsupported_status_is_rejected():
    with pytest.raises(EvidenceValueError):
        EvidenceLedger().add(label="price", value=1.0, status="pending")


def test_derived_entry_requires_known_parents():
    ledger = EvidenceLedger()
    with pytest.raises(UnknownEvidenceError):
        ledger.add(label="change", value=1.0, derived_from=("E1",))


def test_dig_reads_nested_paths():
    payload = {"prices": [{"close": 10.0}, {"close": 11.5}], "currency": "USD"}
    assert dig(payload, "prices.1.close") == 11.5
    assert dig(payload, "currency") == "USD"
    assert dig(payload, "prices.9.close") is None
    assert dig(payload, "missing.deep") is None


def test_add_from_tool_result_records_configured_fields():
    ledger = EvidenceLedger()
    entries = ledger.add_from_tool_result(
        result(), "c2", [FieldSpec("price"), FieldSpec("market_cap", style="compact")]
    )
    assert [entry.label for entry in entries] == ["price", "market_cap"]
    assert entries[0].currency == "USD"
    assert entries[0].tool_call_id == "c2"
    assert entries[0].source == "yfinance"
    assert entries[0].as_of == NOW
    assert entries[1].style == "compact"


def test_unusable_results_record_nothing():
    ledger = EvidenceLedger()
    failed = result(status="error", data=None, error="invalid_ticker")
    assert ledger.add_from_tool_result(failed, "c1", [FieldSpec("price")]) == []
    assert len(ledger) == 0


def test_stale_result_marks_entries_stale():
    ledger = EvidenceLedger()
    entries = ledger.add_from_tool_result(
        result(status="stale"), "c1", [FieldSpec("price")]
    )
    assert entries[0].is_stale is True
    assert ledger.stale_entries() == entries


def test_to_dicts_is_json_ready():
    ledger = EvidenceLedger()
    ledger.add(label="price", value=1.5, currency="USD", as_of=NOW, source="yfinance")
    payload = ledger.to_dicts()[0]
    assert payload["id"] == "E1"
    assert payload["as_of"].startswith("2026-11-09T12:00:00")
    assert payload["derived_from"] == []
