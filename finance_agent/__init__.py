"""Source-layout compatibility shim for local execution.

The implementation lives under ``src/finance_agent`` while this shim keeps
``import finance_agent`` working without requiring an editable install.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "finance_agent")]

from src.finance_agent import (  # noqa: E402
    ADAPTER_PRIMARY,
    ADAPTER_STAGING,
    MODEL_ID,
    RELEASE_NAME,
    RELEASE_REVISION,
)

__all__ = [
    "ADAPTER_PRIMARY",
    "ADAPTER_STAGING",
    "MODEL_ID",
    "RELEASE_NAME",
    "RELEASE_REVISION",
]
