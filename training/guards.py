"""Guards that stop an evaluation from scoring or recording the wrong adapter.

Two failures happened in practice and both were silent:

* ``eval_main`` defaulted to ``/out/adapter-r16-e2-lr0.0001-best`` while the run
  being measured lived under ``...-e3-...``, so the numbers looked plausible and
  described a different model;
* a killed ``modal run`` chain kept executing server-side and wrote its own
  results into the same tag, overwriting the newer artifact.

Kept free of torch and Modal so the rules are unit-testable on a laptop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def assert_adapter(payload: dict[str, Any], adapter: str, env_var: str) -> None:
    """The remote run must report exactly the adapter it was asked to measure."""
    recorded = payload.get("adapter")
    if recorded != adapter:
        raise SystemExit(
            f"{env_var} mismatch: requested {adapter!r} but the run recorded {recorded!r}"
        )


def write_artifact(
    path: Path,
    payload: dict[str, Any],
    adapter: str,
    *,
    allow_overwrite: bool = False,
) -> None:
    """Refuse to replace an artifact that belongs to a different adapter.

    Re-running the same adapter is allowed, because a rerun of the same
    measurement is not an overwrite of different evidence. Reusing a tag for a
    different adapter is exactly how the earlier numbers got mixed, so it fails
    unless the caller opts in explicitly.
    """
    if path.exists() and not allow_overwrite:
        try:
            previous = json.loads(path.read_text(encoding="utf-8")).get("adapter")
        except json.JSONDecodeError as exc:  # a half-written artifact is also unsafe
            raise SystemExit(
                f"refusing to overwrite unreadable artifact {path.name}: {exc}"
            ) from exc
        if previous != adapter:
            raise SystemExit(
                f"refusing to overwrite {path.name}: recorded adapter {previous!r}, "
                f"this run is {adapter!r}. Use a new tag or set ALLOW_ARTIFACT_OVERWRITE=1."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
