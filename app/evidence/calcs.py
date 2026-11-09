"""Registered deterministic calculations over ledger entries (plan section 8.3).

Every derived value records ``derived_from`` so the validator can trace it back
to tool responses. Rules:

* A null input is a hard error - a calculation never invents a substitute.
* Arithmetic across two different currencies is refused, not coerced.
* A derived entry inherits the *oldest* input ``as_of`` and is stale if any
  input was stale, so freshness cannot improve by computing something.
* Percents are expressed in percentage points (``12.5`` means 12.5%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.evidence.ledger import EvidenceEntry, EvidenceLedger

PERCENT_UNIT = "%"


class CalculationError(Exception):
    """Raised when a registered calculation cannot be applied honestly."""


@dataclass(frozen=True, slots=True)
class CalcOutput:
    """Result of a calculation, before it becomes a ledger entry."""

    value: Any
    unit: str | None = None
    currency: str | None = None
    style: str = "plain"


CalcFn = Callable[[EvidenceLedger, tuple[str, ...], dict[str, Any]], CalcOutput]

_CALCS: dict[str, CalcFn] = {}


def register(name: str) -> Callable[[CalcFn], CalcFn]:
    """Register a deterministic calculation under a stable name."""

    def decorator(function: CalcFn) -> CalcFn:
        if name in _CALCS:
            raise ValueError(f"calculation already registered: {name}")
        _CALCS[name] = function
        return function

    return decorator


def available() -> tuple[str, ...]:
    return tuple(sorted(_CALCS))


def get_calc(name: str) -> CalcFn:
    try:
        return _CALCS[name]
    except KeyError as error:
        raise CalculationError(f"unknown calculation: {name}") from error


def apply(
    ledger: EvidenceLedger,
    name: str,
    *evidence_ids: str,
    label: str | None = None,
    **params: Any,
) -> EvidenceEntry:
    """Run a registered calculation and append the result to the ledger."""

    function = get_calc(name)
    inputs = [ledger.get(evidence_id) for evidence_id in evidence_ids]
    output = function(ledger, tuple(evidence_ids), dict(params))

    as_of_values = [entry.as_of for entry in inputs if entry.as_of is not None]
    return ledger.add(
        label=label or name,
        value=output.value,
        tool_call_id=",".join(entry.tool_call_id for entry in inputs if entry.tool_call_id),
        path=f"calc.{name}",
        unit=output.unit,
        currency=output.currency,
        as_of=min(as_of_values) if as_of_values else None,
        source="derived",
        derived_from=tuple(evidence_ids),
        status="stale" if any(entry.is_stale for entry in inputs) else "ok",
        style=output.style,
    )


def _number(entry: EvidenceEntry) -> float:
    if entry.is_null:
        raise CalculationError(f"null evidence cannot be calculated on: {entry.id}")
    if isinstance(entry.value, bool) or not isinstance(entry.value, (int, float)):
        raise CalculationError(f"non-numeric evidence: {entry.id}")
    return float(entry.value)


def _numbers(entry: EvidenceEntry) -> list[float]:
    if entry.is_null:
        raise CalculationError(f"null series cannot be calculated on: {entry.id}")
    if not isinstance(entry.value, (list, tuple)):
        raise CalculationError(f"series evidence must be a list: {entry.id}")
    values: list[float] = []
    for item in entry.value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CalculationError(f"series evidence contains a non-number: {entry.id}")
        values.append(float(item))
    return values


def _require_comparable(left: EvidenceEntry, right: EvidenceEntry) -> None:
    if left.currency and right.currency and left.currency != right.currency:
        raise CalculationError(
            f"currency mismatch: {left.id} is {left.currency}, {right.id} is {right.currency}"
        )
    if left.unit and right.unit and left.unit != right.unit:
        raise CalculationError(f"unit mismatch: {left.id} {left.unit} vs {right.id} {right.unit}")


@register("pct_change")
def pct_change(
    ledger: EvidenceLedger, ids: tuple[str, ...], params: dict[str, Any]
) -> CalcOutput:
    """Percent change from the first entry to the second."""

    if len(ids) != 2:
        raise CalculationError("pct_change expects exactly two evidence ids")
    base_entry, other_entry = (ledger.get(evidence_id) for evidence_id in ids)
    _require_comparable(base_entry, other_entry)
    base, other = _number(base_entry), _number(other_entry)
    if base == 0:
        raise CalculationError(f"percentage change from zero is undefined: {base_entry.id}")
    return CalcOutput(value=(other - base) / base * 100.0, unit=PERCENT_UNIT)


@register("diff")
def diff(ledger: EvidenceLedger, ids: tuple[str, ...], params: dict[str, Any]) -> CalcOutput:
    """Second entry minus first entry, keeping unit and currency."""

    if len(ids) != 2:
        raise CalculationError("diff expects exactly two evidence ids")
    base_entry, other_entry = (ledger.get(evidence_id) for evidence_id in ids)
    _require_comparable(base_entry, other_entry)
    value = _number(other_entry) - _number(base_entry)
    return CalcOutput(
        value=value,
        unit=base_entry.unit,
        currency=base_entry.currency or other_entry.currency,
    )


@register("ratio")
def ratio(ledger: EvidenceLedger, ids: tuple[str, ...], params: dict[str, Any]) -> CalcOutput:
    """First entry divided by the second, dimensionless."""

    if len(ids) != 2:
        raise CalculationError("ratio expects exactly two evidence ids")
    numerator_entry, denominator_entry = (ledger.get(evidence_id) for evidence_id in ids)
    _require_comparable(numerator_entry, denominator_entry)
    denominator = _number(denominator_entry)
    if denominator == 0:
        raise CalculationError(f"division by zero: {denominator_entry.id}")
    return CalcOutput(value=_number(numerator_entry) / denominator)


@register("sma")
def sma(ledger: EvidenceLedger, ids: tuple[str, ...], params: dict[str, Any]) -> CalcOutput:
    """Simple moving average over the trailing ``window`` points of a series."""

    if len(ids) != 1:
        raise CalculationError("sma expects exactly one series evidence id")
    window = params.get("window")
    if not isinstance(window, int) or window < 1:
        raise CalculationError("sma requires a positive integer window")
    series_entry = ledger.get(ids[0])
    values = _numbers(series_entry)
    if len(values) < window:
        raise CalculationError(
            f"sma window {window} exceeds available points ({len(values)}) for {series_entry.id}"
        )
    window_values = values[-window:]
    return CalcOutput(
        value=sum(window_values) / window,
        unit=series_entry.unit,
        currency=series_entry.currency,
    )
