"""The `ToolResult` envelope every finance tool returns (plan section 8.1).

Rules encoded here:
* Tools never raise into the agent loop; failures become ``status`` values.
* ``ok``/``stale`` results must carry data, ``error`` results must carry a reason.
* Timestamps must be timezone-aware so staleness arithmetic cannot silently
  compare naive and aware datetimes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ToolStatus = Literal["ok", "unavailable", "error", "stale"]

#: Statuses that carry usable payloads.
USABLE_STATUSES: frozenset[str] = frozenset({"ok", "stale"})


class ToolResult(BaseModel):
    """Normalized envelope for a single tool invocation."""

    model_config = ConfigDict(extra="forbid")

    status: ToolStatus
    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] | None = None
    source: str = Field(min_length=1)
    retrieved_at: datetime
    as_of: datetime | None = None
    error: str | None = None

    @field_validator("retrieved_at", "as_of")
    @classmethod
    def _require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _check_status_consistency(self) -> "ToolResult":
        if self.status in USABLE_STATUSES and self.data is None:
            raise ValueError(f"{self.status!r} result for {self.tool!r} requires data")
        if self.status == "error" and not self.error:
            raise ValueError(f"'error' result for {self.tool!r} requires an error code")
        return self

    @property
    def usable(self) -> bool:
        """True when the payload may be read by the evidence ledger."""
        return self.status in USABLE_STATUSES and self.data is not None


class ToolCall(BaseModel):
    """One assistant turn of the planning stage: pick a tool, give arguments."""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
