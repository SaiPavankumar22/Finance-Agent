"""Evidence ledger and deterministic grounding checks."""

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class EvidenceLedger:
    """Stores source payloads and numeric values permitted in a report."""

    sources: dict[str, Any] = field(default_factory=dict)
    numeric_values: list[float] = field(default_factory=list)

    def add(self, reference: str, payload: Any) -> None:
        self.sources[reference] = payload
        self.numeric_values.extend(_numbers(payload))

    def references(self) -> list[str]:
        return list(self.sources)


def _numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numbers(item)]
    return []


def validate_numeric_claims(
    claims: Any, ledger: EvidenceLedger, tolerance: float = 1e-6
) -> None:
    """Reject numeric claims that are not present in retrieved evidence."""

    allowed = ledger.numeric_values
    for number in _numbers(claims):
        if not any(math.isclose(number, source, rel_tol=tolerance, abs_tol=tolerance) for source in allowed):
            raise ValueError(f"ungrounded numeric claim: {number}")
