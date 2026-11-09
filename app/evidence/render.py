"""Placeholder rendering: turn ``{{E<n>}}`` into traceable text (plan section 8.4).

Formatting contract - a placeholder renders as a **self-contained quantity**, so
templates read naturally and never need the model to supply units::

    {{E1}}  ->  "189.23 USD"      # price, currency suffix
    {{E2}}  ->  "1.42T USD"       # market cap, compact magnitude
    {{E3}}  ->  "12.50%"          # percent
    {{E4}}  ->  "null"            # missing value (the validator rejects these)
    {{E5}}  ->  "\u201cCompany beats estimates\u201d"   # quoted headline

Order matters for grounding: validate the **template** first, then substitute.
That way every digit in the rendered text necessarily came from a placeholder.
"""

from __future__ import annotations

import re

from app.evidence.ledger import EvidenceEntry, EvidenceLedger, UnknownEvidenceError
from app.schemas.report import Claim, RenderedClaim, ReportSkeleton

PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*(E\d+)\s*\}\}")

_COMPACT_STEPS: tuple[tuple[float, str], ...] = (
    (1e12, "T"),
    (1e9, "B"),
    (1e6, "M"),
    (1e3, "K"),
)


def strip_placeholders(template: str) -> str:
    """Remove well-formed placeholders so the leftover text can be inspected."""

    return PLACEHOLDER_PATTERN.sub("", template)


def referenced_ids(template: str) -> list[str]:
    """Evidence IDs referenced by a template, in first-appearance order, unique."""

    seen: dict[str, None] = {}
    for match in PLACEHOLDER_PATTERN.finditer(template):
        seen.setdefault(match.group(1), None)
    return list(seen)


#: Units that are implied by the sentence and must not be printed.
SILENT_UNITS: frozenset[str] = frozenset({"ratio", "count", "percent"})


def _suffix(entry: EvidenceEntry) -> str:
    if entry.currency:
        return f" {entry.currency}"
    if not entry.unit or entry.unit in SILENT_UNITS:
        return ""
    if entry.unit == "%":
        return "%"
    return f" {entry.unit}"


def _compact(number: float) -> str:
    for scale, symbol in _COMPACT_STEPS:
        if abs(number) >= scale:
            return f"{number / scale:.2f}{symbol}"
    return f"{number:.2f}"


def format_value(entry: EvidenceEntry) -> str:
    """Render one entry's value according to its style, unit and currency."""

    value = entry.value
    if value is None:
        return "null"
    if entry.style == "text":
        return str(value)
    if entry.style == "quote":
        return f"\u201c{value}\u201d"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        if entry.style == "compact":
            return f"{_compact(float(value))}{_suffix(entry)}"
        if isinstance(value, int):
            return f"{value}{_suffix(entry)}"
        return f"{value:.2f}{_suffix(entry)}"
    return str(value)


def render_text(template: str, ledger: EvidenceLedger) -> str:
    """Substitute every placeholder with its ledger value."""

    def replace(match: re.Match[str]) -> str:
        evidence_id = match.group(1)
        try:
            entry = ledger.get(evidence_id)
        except UnknownEvidenceError as error:
            raise UnknownEvidenceError(
                f"template references unknown evidence {evidence_id}"
            ) from error
        return format_value(entry)

    return PLACEHOLDER_PATTERN.sub(replace, template)


def render_claim(claim: Claim, ledger: EvidenceLedger) -> RenderedClaim:
    return RenderedClaim(
        type=claim.type,
        text=render_text(claim.text, ledger),
        evidence=list(claim.evidence),
    )


def render_claims(skeleton: ReportSkeleton, ledger: EvidenceLedger) -> list[RenderedClaim]:
    return [render_claim(claim, ledger) for claim in skeleton.claims]


def format_ledger_line(entry: EvidenceEntry) -> str:
    """Compact, ID-first ledger line shown to the model (plan section 8.3)."""

    kind = "stale" if entry.is_stale else entry.source
    line = f"{entry.id} {entry.label}={format_value(entry)}"
    if entry.as_of is not None:
        line += f" as_of={entry.as_of.isoformat(timespec='seconds').replace('+00:00', 'Z')}"
    if entry.is_derived:
        line += f" derived_from={','.join(entry.derived_from)}"
    return f"{line} [{kind}]"


def render_ledger(ledger: EvidenceLedger) -> str:
    """The whole ledger as a token-cheap block for the composition prompt."""

    return "\n".join(format_ledger_line(entry) for entry in ledger)
