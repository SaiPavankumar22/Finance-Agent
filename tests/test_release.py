import json
from pathlib import Path

from finance_agent import ADAPTER_PRIMARY, ADAPTER_STAGING, RELEASE_NAME

ROOT = Path(__file__).parents[1]


def test_release_metadata_matches_manifest() -> None:
    manifest = json.loads((ROOT / "artifacts/release/ic7.json").read_text())
    assert RELEASE_NAME == manifest["release"]
    assert ADAPTER_PRIMARY == manifest["adapter"]["primary"]
    assert ADAPTER_STAGING == manifest["adapter"]["staging"]


def test_release_manifest_is_accepted() -> None:
    manifest = json.loads((ROOT / "artifacts/release/ic7.json").read_text())
    assert manifest["status"] == "accepted"
    assert manifest["metrics"]["frozen_tool_selection"] >= 0.80
    assert manifest["metrics"]["india_tool_selection"] >= 0.70
