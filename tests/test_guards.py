"""Guards that stop an evaluation recording the wrong adapter."""

import json

import pytest

from training.guards import assert_adapter, write_artifact


def test_assert_adapter_rejects_a_mismatched_run():
    payload = {"adapter": "/out/adapter-r16-e2-lr0.0001-best"}
    with pytest.raises(SystemExit) as error:
        assert_adapter(payload, "/out/adapter-r16-e4-lr0.0001-best", "SFT_ADAPTER")
    assert "SFT_ADAPTER mismatch" in str(error.value)


def test_assert_adapter_accepts_the_requested_adapter():
    payload = {"adapter": "/out/adapter-r16-e4-lr0.0001-best"}
    assert_adapter(payload, "/out/adapter-r16-e4-lr0.0001-best", "SFT_ADAPTER")


def test_write_artifact_refuses_to_replace_another_adapters_artifact(tmp_path):
    path = tmp_path / "eval-ic4.json"
    write_artifact(
        path,
        {"adapter": "/out/adapter-r16-e3-lr0.0001-best", "eval": 1},
        "/out/adapter-r16-e3-lr0.0001-best",
    )
    with pytest.raises(SystemExit) as error:
        write_artifact(
            path,
            {"adapter": "/out/adapter-r16-e4-lr0.0001-best"},
            "/out/adapter-r16-e4-lr0.0001-best",
        )
    assert "refusing to overwrite" in str(error.value)
    assert json.loads(path.read_text(encoding="utf-8"))["eval"] == 1


def test_write_artifact_allows_a_rerun_of_the_same_adapter(tmp_path):
    path = tmp_path / "eval-ic4.json"
    adapter = "/out/adapter-r16-e4-lr0.0001-best"
    write_artifact(path, {"adapter": adapter, "eval": 1}, adapter)
    write_artifact(path, {"adapter": adapter, "eval": 2}, adapter)
    assert json.loads(path.read_text(encoding="utf-8"))["eval"] == 2


def test_write_artifact_allows_an_explicit_override(tmp_path):
    path = tmp_path / "eval-ic4.json"
    write_artifact(path, {"adapter": "/out/a", "eval": 1}, "/out/a")
    write_artifact(path, {"adapter": "/out/b", "eval": 2}, "/out/b", allow_overwrite=True)
    assert json.loads(path.read_text(encoding="utf-8"))["eval"] == 2
