"""Evidence ledger: every retrieved value gets a stable ID (plan section 8.3).

An entry looks like::

    {"id": "E7", "tool_call_id": "c2", "path": "price", "value": 189.23,
     "unit": null, "currency": "USD", "as_of": "...", "source": "yfinance"}

IDs are assigned monotonically (``E1``, ``E2``, ...) and never reused, so a
report that cites ``E7`` cites the same value forever. Null values are recorded
too: a missing ratio is evidence that the ratio is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Iterator, Mapping

from app.schemas.tools import ToolResult

#: ``"ok"`` or ``"stale"`` - stale entries are still real, just old.
EntryStatus = str

#: Rendering hints. ``plain`` keeps full precision to two decimals, ``compact``
#: scales large magnitudes (1.42T), ``text`` passes a string through, ``quote``
#: wraps a string in curly quotes.
ValueStyle = str


class LedgerError(Exception):
    """Base class for ledger failures."""


class UnknownEvidenceError(LedgerError):
    """Raised when an evidence ID is not present in the ledger."""


class EvidenceValueError(LedgerError):
    """Raised when an entry cannot be constructed."""


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Which field of a tool payload becomes evidence, and how it is labelled."""

    path: str
    label: str | None = None
    unit: str | None = None
    currency_path: str | None = "currency"
    style: ValueStyle = "plain"

    @property
    def resolved_label(self) -> str:
        if self.label:
            return self.label
        return self.path.rsplit(".", 1)[-1]


@dataclass(frozen=True, slots=True)
class EvidenceEntry:
    """One grounded value."""

    id: str
    label: str
    value: Any
    tool_call_id: str
    path: str
    unit: str | None = None
    currency: str | None = None
    as_of: datetime | None = None
    source: str = "unknown"
    derived_from: tuple[str, ...] = ()
    status: EntryStatus = "ok"
    style: ValueStyle = "plain"

    @property
    def is_null(self) -> bool:
        return self.value is None

    @property
    def is_derived(self) -> bool:
        return bool(self.derived_from)

    @property
    def is_stale(self) -> bool:
        return self.status == "stale"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "value": self.value,
            "tool_call_id": self.tool_call_id,
            "path": self.path,
            "unit": self.unit,
            "currency": self.currency,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "source": self.source,
            "derived_from": list(self.derived_from),
            "status": self.status,
            "style": self.style,
        }


def dig(payload: Any, path: str) -> Any:
    """Read a dotted path (``a.b.0``) out of a nested mapping/list payload."""

    current = payload
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, (list, tuple)):
            if not part.isdigit():
                return None
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            return None
    return current


class EvidenceLedger:
    """Ordered collection of grounded values with stable IDs."""

    def __init__(self) -> None:
        self._entries: dict[str, EvidenceEntry] = {}
        self._counter = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[EvidenceEntry]:
        return iter(self._entries.values())

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._entries

    @property
    def entries(self) -> tuple[EvidenceEntry, ...]:
        return tuple(self._entries.values())

    def ids(self) -> list[str]:
        return list(self._entries)

    def get(self, evidence_id: str) -> EvidenceEntry:
        try:
            return self._entries[evidence_id]
        except KeyError as error:
            raise UnknownEvidenceError(f"unknown evidence id: {evidence_id}") from error

    def add(
        self,
        *,
        label: str,
        value: Any,
        tool_call_id: str = "",
        path: str = "",
        unit: str | None = None,
        currency: str | None = None,
        as_of: datetime | None = None,
        source: str = "unknown",
        derived_from: Iterable[str] = (),
        status: EntryStatus = "ok",
        style: ValueStyle = "plain",
    ) -> EvidenceEntry:
        """Append one value and return its entry."""

        if not label:
            raise EvidenceValueError("evidence entries require a label")
        if status not in {"ok", "stale"}:
            raise EvidenceValueError(f"unsupported entry status: {status!r}")
        parents = tuple(derived_from)
        for parent in parents:
            if parent not in self._entries:
                raise UnknownEvidenceError(f"derived entry references unknown id: {parent}")

        self._counter += 1
        entry = EvidenceEntry(
            id=f"E{self._counter}",
            label=label,
            value=value,
            tool_call_id=tool_call_id,
            path=path,
            unit=unit,
            currency=currency,
            as_of=as_of,
            source=source,
            derived_from=parents,
            status=status,
            style=style,
        )
        self._entries[entry.id] = entry
        return entry

    def add_from_tool_result(
        self,
        result: ToolResult,
        tool_call_id: str,
        fields: Iterable[FieldSpec],
    ) -> list[EvidenceEntry]:
        """Record the configured fields of a tool result.

        Unusable results record nothing: the absence itself is reported by the
        planner through ``missing``, not invented as a zero.
        """

        if not result.usable or result.data is None:
            return []

        entry_status = "stale" if result.status == "stale" else "ok"
        added: list[EvidenceEntry] = []
        for spec in fields:
            currency = None
            if spec.currency_path:
                raw_currency = dig(result.data, spec.currency_path)
                currency = raw_currency if isinstance(raw_currency, str) else None
            added.append(
                self.add(
                    label=spec.resolved_label,
                    value=dig(result.data, spec.path),
                    tool_call_id=tool_call_id,
                    path=spec.path,
                    unit=spec.unit,
                    currency=currency,
                    as_of=result.as_of,
                    source=result.source,
                    status=entry_status,
                    style=spec.style,
                )
            )
        return added

    def null_entries(self) -> list[EvidenceEntry]:
        return [entry for entry in self if entry.is_null]

    def stale_entries(self) -> list[EvidenceEntry]:
        return [entry for entry in self if entry.is_stale]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self]
