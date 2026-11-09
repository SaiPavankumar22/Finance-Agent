"""Rendering must produce self-contained quantities and never leak braces."""

from datetime import datetime, timezone

import pytest

from app.evidence.ledger import EvidenceLedger, UnknownEvidenceError
from app.evidence.render import (
    format_ledger_line,
    format_value,
    referenced_ids,
    render_claims,
    render_ledger,
    render_text,
    strip_placeholders,
)
from app.schemas.report import Claim, ReportSkeleton

NOW = datetime(2026, 11, 9, 9, 30, tzinfo=timezone.utc)


@pytest.fixture
def ledger():
    instance = EvidenceLedger()
    instance.add(label="price", value=189.23, currency="USD", as_of=NOW, source="yfinance")
    instance.add(label="market_cap", value=2.9e12, currency="USD", style="compact")
    instance.add(label="volume", value=44_703_806, unit="shares", style="compact")
    instance.add(label="pe_ratio", value=28.4, unit="ratio")
    instance.add(label="change", value=12.5, unit="%")
    instance.add(label="ev_ebitda", value=None, unit="ratio")
    instance.add(label="headline_1", value="Company beats estimates", style="quote")
    instance.add(label="observations", value=252, unit="count", style="plain")
    return instance


def test_plain_value_carries_currency(ledger):
    assert format_value(ledger.get("E1")) == "189.23 USD"


def test_compact_value_scales_magnitude(ledger):
    assert format_value(ledger.get("E2")) == "2.90T USD"
    assert format_value(ledger.get("E3")) == "44.70M shares"


def test_ratio_renders_without_a_unit_word(ledger):
    assert format_value(ledger.get("E4")) == "28.40"


def test_percent_renders_with_symbol(ledger):
    assert format_value(ledger.get("E5")) == "12.50%"


def test_null_renders_as_null(ledger):
    assert format_value(ledger.get("E6")) == "null"


def test_quote_style_wraps_headline(ledger):
    assert format_value(ledger.get("E7")) == "\u201cCompany beats estimates\u201d"


def test_integer_count_has_no_suffix(ledger):
    assert format_value(ledger.get("E8")) == "252"


def test_render_text_substitutes_values(ledger):
    text = render_text("The price is {{E1}} and the market cap is {{E2}}.", ledger)
    assert text == "The price is 189.23 USD and the market cap is 2.90T USD."
    assert "{{" not in text and "}}" not in text


def test_render_text_accepts_spaced_placeholders(ledger):
    assert render_text("Price: {{ E1 }}", ledger) == "Price: 189.23 USD"


def test_render_text_rejects_unknown_placeholder(ledger):
    with pytest.raises(UnknownEvidenceError):
        render_text("Price: {{E42}}", ledger)


def test_referenced_ids_are_ordered_and_unique(ledger):
    assert referenced_ids("{{E2}} then {{E1}} then {{E2}}") == ["E2", "E1"]


def test_strip_placeholders_removes_only_well_formed_ones(ledger):
    assert strip_placeholders("a {{E1}} b") == "a  b"
    assert "{{" in strip_placeholders("a {{E1 b")


def test_ledger_line_is_id_first(ledger):
    line = format_ledger_line(ledger.get("E1"))
    assert line.startswith("E1 price=189.23 USD")
    assert "as_of=2026-11-09T09:30:00Z" in line
    assert line.endswith("[yfinance]")


def test_stale_entries_are_labelled_in_the_ledger_text():
    ledger = EvidenceLedger()
    ledger.add(label="price", value=1.0, currency="USD", status="stale", source="yfinance")
    assert render_ledger(ledger) == "E1 price=1.00 USD [stale]"


def test_render_claims_renders_every_claim(ledger):
    skeleton = ReportSkeleton(
        claims=[
            Claim(type="fact", text="Price is {{E1}}.", evidence=["E1"]),
            Claim(type="interpretation", text="Valuation looks full.", evidence=["E4"]),
        ]
    )
    rendered = render_claims(skeleton, ledger)
    assert [claim.text for claim in rendered] == ["Price is 189.23 USD.", "Valuation looks full."]
    assert rendered[0].evidence == ["E1"]
