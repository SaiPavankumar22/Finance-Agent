"""Report schemas: what the model emits (skeleton) and what code emits (composed).

The model writes claim text containing ``{{E<n>}}`` placeholders and no digits
(plan section 8.4). Code substitutes ledger values afterwards, so every digit in
the final text is traceable to a tool response or a registered calculation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ClaimType = Literal["fact", "calculation", "interpretation"]

DISCLAIMER = "This is informational research, not personalized investment advice."


class Claim(BaseModel):
    """One sentence of the report, with evidence references."""

    model_config = ConfigDict(extra="forbid")

    type: ClaimType
    text: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)


class MissingItem(BaseModel):
    """A datum the agent could not obtain, and why."""

    model_config = ConfigDict(extra="forbid")

    item: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ReportSkeleton(BaseModel):
    """Stage-2 model output, pre-substitution."""

    model_config = ConfigDict(extra="forbid")

    claims: list[Claim] = Field(default_factory=list)
    missing: list[MissingItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class RenderedClaim(BaseModel):
    """A claim whose placeholders have been replaced with ledger values."""

    model_config = ConfigDict(extra="forbid")

    type: ClaimType
    text: str
    evidence: list[str] = Field(default_factory=list)


class EvidenceStamp(BaseModel):
    """Provenance line shown next to a report: what, from where, as of when."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    label: str
    source: str
    as_of: datetime | None = None


class ComposedReport(BaseModel):
    """Final structured report returned by the API."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "2.0"
    question: str
    claims: list[RenderedClaim] = Field(default_factory=list)
    missing: list[MissingItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence: list[EvidenceStamp] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER
    generated_at: datetime
    model_version: str | None = None
    dataset_version: str | None = None
    code_sha: str | None = None
