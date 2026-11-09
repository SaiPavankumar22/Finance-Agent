"""Deterministic grounding checks for stage-2 report skeletons (plan section 8.4).

The validator treats the model's output as untrusted text and enforces:

1. No digits outside ``{{E<n>}}`` placeholders (this is the whole point: the
   model never writes a number).
2. No malformed or unknown placeholders.
3. Every placeholder is declared in the claim's ``evidence`` list, and every
   declared evidence ID exists in the ledger.
4. ``fact``/``calculation`` claims cite evidence, and no cited value is null or
   (by policy) stale.
5. Claims do not mix currencies, or incomparable units.

It returns a :class:`ValidationReport` rather than raising, so callers can log
what was removed; :func:`enforce` raises for pipeline stages that must fail fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.evidence.ledger import EvidenceEntry, EvidenceLedger
from app.evidence.render import PLACEHOLDER_PATTERN, referenced_ids, strip_placeholders
from app.schemas.report import Claim, ReportSkeleton

#: Units that compose freely with anything else inside one sentence.
COMPOSABLE_UNITS: frozenset[str] = frozenset({"%", "percent", "ratio"})


@dataclass(frozen=True, slots=True)
class Issue:
    """One grounding violation."""

    code: str
    message: str
    claim_index: int | None = None
    evidence_id: str | None = None


@dataclass
class ValidationReport:
    """Collected issues for one skeleton."""

    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def codes(self) -> list[str]:
        return [issue.code for issue in self.issues]

    def has(self, code: str) -> bool:
        return any(issue.code == code for issue in self.issues)

    def messages(self) -> list[str]:
        return [issue.message for issue in self.issues]


class ValidationError(Exception):
    """Raised by :func:`enforce` when a skeleton fails grounding."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("; ".join(report.messages()) or "validation failed")


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    """Knobs for behaviours the plan leaves to a documented rule."""

    allow_stale_facts: bool = False
    allow_null_evidence: bool = False
    require_declared_evidence: bool = True


DEFAULT_POLICY = ValidationPolicy()


def _contains_digit(text: str) -> bool:
    return any(character.isdigit() for character in text)


def _claimed_entries(claim: Claim, ledger: EvidenceLedger) -> list[EvidenceEntry]:
    entries: list[EvidenceEntry] = []
    for evidence_id in claim.evidence:
        if evidence_id in ledger:
            entries.append(ledger.get(evidence_id))
    return entries


def _check_text(index: int, claim: Claim) -> list[Issue]:
    issues: list[Issue] = []
    leftover = strip_placeholders(claim.text)
    if _contains_digit(leftover):
        digits = "".join(character for character in leftover if character.isdigit())
        issues.append(
            Issue(
                "digit_outside_placeholder",
                f"claim {index} writes digits in prose: {digits!r}",
                claim_index=index,
            )
        )
    if "{{" in leftover or "}}" in leftover:
        issues.append(
            Issue(
                "malformed_placeholder",
                f"claim {index} contains an unparseable placeholder",
                claim_index=index,
            )
        )
    if not leftover.strip():
        issues.append(
            Issue(
                "empty_claim_text",
                f"claim {index} has no text outside placeholders",
                claim_index=index,
            )
        )
    return issues


def _check_references(index: int, claim: Claim, ledger: EvidenceLedger) -> list[Issue]:
    issues: list[Issue] = []
    placeholders = referenced_ids(claim.text)
    for evidence_id in placeholders:
        if evidence_id not in ledger:
            issues.append(
                Issue(
                    "unknown_placeholder",
                    f"claim {index} references unknown evidence {evidence_id}",
                    claim_index=index,
                    evidence_id=evidence_id,
                )
            )
    for evidence_id in claim.evidence:
        if evidence_id not in ledger:
            issues.append(
                Issue(
                    "unknown_evidence",
                    f"claim {index} declares evidence {evidence_id} that is not in the ledger",
                    claim_index=index,
                    evidence_id=evidence_id,
                )
            )
    return issues


def _check_declarations(index: int, claim: Claim, policy: ValidationPolicy) -> list[Issue]:
    if not policy.require_declared_evidence:
        return []
    issues: list[Issue] = []
    declared = set(claim.evidence)
    for evidence_id in referenced_ids(claim.text):
        if evidence_id not in declared:
            issues.append(
                Issue(
                    "undeclared_evidence",
                    f"claim {index} uses {evidence_id} without declaring it",
                    claim_index=index,
                    evidence_id=evidence_id,
                )
            )
    return issues


def _check_grounding(
    index: int, claim: Claim, ledger: EvidenceLedger, policy: ValidationPolicy
) -> list[Issue]:
    if claim.type == "interpretation":
        entries = _claimed_entries(claim, ledger)
    else:
        if not claim.evidence:
            return [
                Issue(
                    "ungrounded_claim",
                    f"claim {index} is a {claim.type} claim with no evidence",
                    claim_index=index,
                )
            ]
        entries = _claimed_entries(claim, ledger)

    issues: list[Issue] = []
    for entry in entries:
        if entry.is_null and not policy.allow_null_evidence:
            issues.append(
                Issue(
                    "null_evidence",
                    f"claim {index} cites {entry.id} ({entry.label}) which is unavailable",
                    claim_index=index,
                    evidence_id=entry.id,
                )
            )
        if entry.is_stale and claim.type != "interpretation" and not policy.allow_stale_facts:
            issues.append(
                Issue(
                    "stale_evidence",
                    f"claim {index} cites {entry.id} ({entry.label}) which is stale",
                    claim_index=index,
                    evidence_id=entry.id,
                )
            )
    return issues


def _check_units(index: int, claim: Claim, ledger: EvidenceLedger) -> list[Issue]:
    entries = _claimed_entries(claim, ledger)
    issues: list[Issue] = []
    currencies = {entry.currency for entry in entries if entry.currency}
    if len(currencies) > 1:
        issues.append(
            Issue(
                "currency_mismatch",
                f"claim {index} mixes currencies: {', '.join(sorted(currencies))}",
                claim_index=index,
            )
        )
    units = {
        entry.unit
        for entry in entries
        if entry.unit and entry.unit not in COMPOSABLE_UNITS
    }
    if len(units) > 1:
        issues.append(
            Issue(
                "unit_mismatch",
                f"claim {index} mixes units: {', '.join(sorted(units))}",
                claim_index=index,
            )
        )
    return issues


def validate_skeleton(
    skeleton: ReportSkeleton,
    ledger: EvidenceLedger,
    policy: ValidationPolicy = DEFAULT_POLICY,
) -> ValidationReport:
    """Check every claim in a skeleton against the ledger."""

    issues: list[Issue] = []
    for index, claim in enumerate(skeleton.claims):
        issues.extend(_check_text(index, claim))
        issues.extend(_check_references(index, claim, ledger))
        issues.extend(_check_declarations(index, claim, policy))
        issues.extend(_check_grounding(index, claim, ledger, policy))
        issues.extend(_check_units(index, claim, ledger))

    if not skeleton.claims and not skeleton.missing:
        issues.append(
            Issue(
                "empty_report",
                "a report must contain claims or explain what is missing",
            )
        )
    return ValidationReport(issues)


def enforce(
    skeleton: ReportSkeleton,
    ledger: EvidenceLedger,
    policy: ValidationPolicy = DEFAULT_POLICY,
) -> ReportSkeleton:
    """Validate and return the skeleton, or raise :class:`ValidationError`."""

    report = validate_skeleton(skeleton, ledger, policy)
    if not report.ok:
        raise ValidationError(report)
    return skeleton


def is_grounded(skeleton: ReportSkeleton, ledger: EvidenceLedger) -> bool:
    """Convenience predicate used by pipelines and tests."""

    return validate_skeleton(skeleton, ledger).ok


__all__ = [
    "DEFAULT_POLICY",
    "Issue",
    "ValidationError",
    "ValidationPolicy",
    "ValidationReport",
    "enforce",
    "is_grounded",
    "validate_skeleton",
    "PLACEHOLDER_PATTERN",
]
