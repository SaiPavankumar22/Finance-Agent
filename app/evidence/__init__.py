"""Evidence layer: ledger, registered calculations, renderer and validator."""

from __future__ import annotations

from app.evidence import calcs, ledger, render, validator
from app.evidence.calcs import CalcOutput, CalculationError
from app.evidence.calcs import apply as apply_calc
from app.evidence.ledger import (
    EvidenceEntry,
    EvidenceLedger,
    FieldSpec,
    LedgerError,
    UnknownEvidenceError,
    dig,
)
from app.evidence.render import (
    format_ledger_line,
    format_value,
    referenced_ids,
    render_claims,
    render_ledger,
    render_text,
)
from app.evidence.validator import (
    DEFAULT_POLICY,
    Issue,
    ValidationError,
    ValidationPolicy,
    ValidationReport,
    enforce,
    is_grounded,
    validate_skeleton,
)

__all__ = [
    "DEFAULT_POLICY",
    "CalculationError",
    "CalcOutput",
    "EvidenceEntry",
    "EvidenceLedger",
    "FieldSpec",
    "Issue",
    "LedgerError",
    "UnknownEvidenceError",
    "ValidationError",
    "ValidationPolicy",
    "ValidationReport",
    "apply_calc",
    "calcs",
    "dig",
    "enforce",
    "format_ledger_line",
    "format_value",
    "is_grounded",
    "ledger",
    "referenced_ids",
    "render",
    "render_claims",
    "render_ledger",
    "render_text",
    "validate_skeleton",
    "validator",
]
