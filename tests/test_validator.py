"""Grounding checks: the model must never be able to smuggle a number past code."""

from datetime import datetime, timezone

import pytest

from app.evidence.ledger import EvidenceLedger
from app.evidence.render import render_claims
from app.evidence.validator import (
    ValidationError,
    ValidationPolicy,
    enforce,
    is_grounded,
    validate_skeleton,
)
from app.schemas.report import Claim, MissingItem, ReportSkeleton

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def ledger():
    instance = EvidenceLedger()
    instance.add(label="price", value=189.23, currency="USD", as_of=NOW, source="yfinance")
    instance.add(
        label="market_cap", value=2.9e12, currency="USD", source="yfinance", style="compact"
    )
    instance.add(label="pe_ratio", value=28.4, unit="ratio", source="yfinance")
    instance.add(label="ev_ebitda", value=None, unit="ratio", source="yfinance")
    instance.add(label="price_eur", value=170.0, currency="EUR", source="yfinance")
    instance.add(label="old_price", value=150.0, currency="USD", status="stale")
    instance.add(label="shares_traded", value=1000, unit="shares", source="yfinance")
    instance.add(label="headline_count", value=5, unit="count", source="yfinance")
    instance.add(label="change_pct", value=12.5, unit="%", source="derived")
    return instance


def skeleton(*claims, missing=None):
    return ReportSkeleton(
        claims=list(claims),
        missing=missing or [],
        limitations=["Data from a single provider."],
    )


def fact(text, evidence):
    return Claim(type="fact", text=text, evidence=evidence)


def test_valid_report_passes(ledger):
    report = validate_skeleton(
        skeleton(
            fact("The price is {{E1}} and the market cap is {{E2}}.", ["E1", "E2"]),
            Claim(
                type="interpretation",
                text="Valuation looks full relative to the balance sheet.",
                evidence=["E3"],
            ),
        ),
        ledger,
    )
    assert report.ok is True, report.messages()


def test_digit_written_in_prose_is_rejected(ledger):
    report = validate_skeleton(skeleton(fact("The price is 189.23 USD.", ["E1"])), ledger)
    assert report.has("digit_outside_placeholder")


def test_percentage_written_in_prose_is_rejected(ledger):
    report = validate_skeleton(skeleton(fact("Shares rose 12% this quarter.", ["E9"])), ledger)
    assert report.has("digit_outside_placeholder")


def test_decimal_smuggled_into_prose_is_rejected(ledger):
    report = validate_skeleton(skeleton(fact("It trades at roughly 0.5 times book.", [])), ledger)
    assert report.has("digit_outside_placeholder")


def test_non_latin_digits_are_rejected(ledger):
    report = validate_skeleton(skeleton(fact("السعر هو ١٨٩ دولار.", ["E1"])), ledger)
    assert report.has("digit_outside_placeholder")


def test_digits_inside_a_placeholder_are_allowed(ledger):
    report = validate_skeleton(skeleton(fact("Price: {{E1}}.", ["E1"])), ledger)
    assert report.ok is True, report.messages()


def test_unknown_placeholder_is_rejected(ledger):
    report = validate_skeleton(skeleton(fact("Price: {{E99}}.", ["E99"])), ledger)
    assert report.has("unknown_placeholder")
    assert report.has("unknown_evidence")


def test_malformed_placeholder_is_rejected(ledger):
    report = validate_skeleton(skeleton(fact("Price: {{E1.", ["E1"])), ledger)
    assert report.has("malformed_placeholder")


def test_bare_brace_placeholder_is_rejected(ledger):
    report = validate_skeleton(skeleton(fact("Price: {E1}.", ["E1"])), ledger)
    assert report.ok is False


def test_undeclared_placeholder_is_rejected(ledger):
    report = validate_skeleton(skeleton(fact("Price: {{E1}}.", [])), ledger)
    assert report.has("undeclared_evidence")


def test_declared_evidence_missing_from_ledger_is_rejected(ledger):
    report = validate_skeleton(skeleton(fact("Price: {{E1}}.", ["E1", "E77"])), ledger)
    assert report.has("unknown_evidence")


def test_null_evidence_cannot_be_quoted_as_a_fact(ledger):
    report = validate_skeleton(skeleton(fact("EV/EBITDA stands at {{E4}}.", ["E4"])), ledger)
    assert report.has("null_evidence")


def test_stale_evidence_is_rejected_by_default(ledger):
    report = validate_skeleton(skeleton(fact("The old price was {{E6}}.", ["E6"])), ledger)
    assert report.has("stale_evidence")


def test_policy_can_allow_stale_facts(ledger):
    report = validate_skeleton(
        skeleton(fact("The earlier price was {{E6}}.", ["E6"])),
        ledger,
        ValidationPolicy(allow_stale_facts=True),
    )
    assert report.ok is True, report.messages()


def test_mixed_currencies_are_rejected(ledger):
    report = validate_skeleton(
        skeleton(fact("It trades at {{E1}} versus {{E5}} abroad.", ["E1", "E5"])), ledger
    )
    assert report.has("currency_mismatch")


def test_mixed_incomparable_units_are_rejected(ledger):
    report = validate_skeleton(
        skeleton(fact("Volume {{E7}} across {{E8}} stories.", ["E7", "E8"])), ledger
    )
    assert report.has("unit_mismatch")


def test_percent_and_currency_can_share_a_claim(ledger):
    report = validate_skeleton(
        skeleton(fact("At {{E1}} the move was {{E9}}.", ["E1", "E9"])), ledger
    )
    assert report.ok is True, report.messages()


def test_fact_without_evidence_is_rejected(ledger):
    report = validate_skeleton(skeleton(fact("The company is undervalued.", [])), ledger)
    assert report.has("ungrounded_claim")


def test_interpretation_without_evidence_is_allowed(ledger):
    report = validate_skeleton(
        skeleton(Claim(type="interpretation", text="Momentum looks constructive.", evidence=[])),
        ledger,
    )
    assert report.ok is True, report.messages()


def test_claim_text_that_is_only_a_placeholder_is_rejected(ledger):
    report = validate_skeleton(skeleton(fact("{{E1}}", ["E1"])), ledger)
    assert report.has("empty_claim_text")


def test_repeating_the_same_placeholder_is_allowed(ledger):
    report = validate_skeleton(
        skeleton(fact("{{E1}} is the price, and {{E1}} is repeated.", ["E1"])), ledger
    )
    assert report.ok is True, report.messages()


def test_empty_report_is_rejected(ledger):
    report = validate_skeleton(ReportSkeleton(claims=[], missing=[]), ledger)
    assert report.has("empty_report")


def test_abstention_with_missing_items_passes(ledger):
    report = validate_skeleton(
        skeleton(
            missing=[MissingItem(item="quote", reason="provider_unavailable")],
        ),
        ledger,
    )
    assert report.ok is True, report.messages()


def test_enforce_raises_with_the_report(ledger):
    with pytest.raises(ValidationError) as error:
        enforce(skeleton(fact("The price is 1.00 USD.", [])), ledger)
    assert error.value.report.has("digit_outside_placeholder")


def test_is_grounded_matches_the_report(ledger):
    good = skeleton(fact("Price {{E1}}.", ["E1"]))
    bad = skeleton(fact("Price 189.23 USD.", []))
    assert is_grounded(good, ledger) is True
    assert is_grounded(bad, ledger) is False


def test_validated_then_rendered_text_has_no_brace_or_invented_digits(ledger):
    report = skeleton(fact("Price {{E1}} with cap {{E2}}.", ["E1", "E2"]))
    enforce(report, ledger)
    rendered = render_claims(report, ledger)[0].text
    assert "{{" not in rendered
    assert "189.23 USD" in rendered
    assert "2.90T USD" in rendered
