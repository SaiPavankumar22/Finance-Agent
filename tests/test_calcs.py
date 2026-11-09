"""Registered calculations must refuse to produce dishonest numbers."""

from datetime import datetime, timedelta, timezone

import pytest

from app.evidence import calcs
from app.evidence.calcs import CalculationError
from app.evidence.calcs import apply as apply_calc
from app.evidence.ledger import EvidenceLedger

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


def seeded():
    ledger = EvidenceLedger()
    ledger.add(label="price", value=100.0, currency="USD", as_of=NOW, source="yfinance")
    ledger.add(
        label="old_price",
        value=90.0,
        currency="USD",
        as_of=NOW - timedelta(days=30),
        source="yfinance",
    )
    ledger.add(label="price_eur", value=80.0, currency="EUR", as_of=NOW, source="yfinance")
    ledger.add(label="volume", value=1000.0, unit="shares", as_of=NOW, source="yfinance")
    ledger.add(label="pe", value=25.0, unit="ratio", as_of=NOW, source="yfinance")
    ledger.add(label="closes", value=[10.0, 20.0, 30.0, 40.0], currency="USD")
    return ledger


def test_pct_change_is_in_percentage_points():
    ledger = seeded()
    entry = apply_calc(ledger, "pct_change", "E1", "E2", label="price_change")
    assert entry.value == pytest.approx(-10.0)
    assert entry.unit == "%"
    assert entry.currency is None
    assert entry.derived_from == ("E1", "E2")
    assert entry.source == "derived"


def test_diff_keeps_currency_and_unit():
    ledger = seeded()
    entry = apply_calc(ledger, "diff", "E1", "E2")
    assert entry.value == pytest.approx(-10.0)
    assert entry.currency == "USD"


def test_ratio_is_dimensionless():
    ledger = seeded()
    entry = apply_calc(ledger, "ratio", "E1", "E2")
    assert entry.value == pytest.approx(100.0 / 90.0)
    assert entry.unit is None and entry.currency is None


def test_sma_uses_trailing_window():
    ledger = seeded()
    entry = apply_calc(ledger, "sma", "E6", window=2)
    assert entry.value == pytest.approx(35.0)
    assert entry.currency == "USD"


def test_sma_window_longer_than_series_is_refused():
    ledger = seeded()
    with pytest.raises(CalculationError):
        apply_calc(ledger, "sma", "E6", window=9)


def test_cross_currency_arithmetic_is_refused():
    ledger = seeded()
    with pytest.raises(CalculationError):
        apply_calc(ledger, "diff", "E1", "E3")


def test_unit_mismatch_is_refused():
    ledger = seeded()
    with pytest.raises(CalculationError):
        apply_calc(ledger, "ratio", "E5", "E4")


def test_null_input_is_refused():
    ledger = seeded()
    ledger.add(label="ev_ebitda", value=None, unit="ratio")
    with pytest.raises(CalculationError):
        apply_calc(ledger, "pct_change", "E1", "E7")


def test_zero_denominator_is_refused():
    ledger = EvidenceLedger()
    ledger.add(label="a", value=1.0)
    ledger.add(label="zero", value=0.0)
    with pytest.raises(CalculationError):
        apply_calc(ledger, "ratio", "E1", "E2")


def test_unknown_calculation_is_refused():
    with pytest.raises(CalculationError):
        apply_calc(seeded(), "magic", "E1")


def test_derived_as_of_is_the_oldest_input():
    ledger = seeded()
    entry = apply_calc(ledger, "pct_change", "E1", "E2")
    assert entry.as_of == NOW - timedelta(days=30)


def test_derived_entry_inherits_staleness():
    ledger = EvidenceLedger()
    ledger.add(label="fresh", value=10.0, as_of=NOW)
    ledger.add(label="old", value=11.0, as_of=NOW, status="stale")
    entry = apply_calc(ledger, "pct_change", "E1", "E2")
    assert entry.is_stale is True


def test_registry_lists_the_named_calculations():
    assert {"pct_change", "diff", "ratio", "sma"} <= set(calcs.available())
