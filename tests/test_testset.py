"""The frozen test set must stay frozen, replayable and behaviour-consistent."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.tools.replay import FixtureStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTSET = PROJECT_ROOT / "eval" / "testset.jsonl"
HASH_FILE = PROJECT_ROOT / "eval" / "testset.sha256"
META_FILE = PROJECT_ROOT / "eval" / "testset.meta.json"
REQUIRED_INTENTS = {
    "overview",
    "metric_pe",
    "comparison",
    "non_us_currency",
    "invalid_ticker",
    "missing_ratios",
    "news_unavailable",
    "stale_data",
    "malformed_error",
    "period_ambiguity",
    "unsupported_metric",
    "advice_request",
    "loop_trap",
}

_spec = importlib.util.spec_from_file_location(
    "inject_failures", PROJECT_ROOT / "pipelines" / "02_inject_failures.py"
)
injections = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = injections  # dataclasses need the module registered
_spec.loader.exec_module(injections)

pytestmark = pytest.mark.skipif(
    not TESTSET.exists(), reason="test set has not been generated yet"
)


def examples():
    return [json.loads(line) for line in TESTSET.read_text(encoding="utf-8").splitlines()]


def test_hash_matches_the_frozen_digest():
    payload = TESTSET.read_bytes()
    frozen = HASH_FILE.read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(payload).hexdigest() == frozen


def test_meta_matches_the_file():
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    rows = examples()
    assert meta["count"] == len(rows)
    assert meta["sha256"] == HASH_FILE.read_text(encoding="utf-8").split()[0]
    assert len(rows) >= 120


def test_ids_and_questions_are_unique():
    rows = examples()
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["question"] for row in rows}) == len(rows)


def test_every_intent_is_covered():
    present = {row["intent"] for row in examples()}
    assert REQUIRED_INTENTS <= present


def test_mutations_match_their_expected_behaviour():
    for row in examples():
        mutation = row["mutation"]
        if mutation == "none":
            continue
        assert mutation in injections.MUTATIONS
        assert row["expected_behavior"] == injections.MUTATIONS[mutation].expected_behavior


def test_every_example_is_replayable_from_fixtures():
    store = FixtureStore()
    for row in examples():
        for key, status in row["fixture_status"].items():
            tool, ticker = key.split("@", 1)
            period = row["period"] if tool == "get_historical_prices" else None
            assert store.has(tool, ticker, period), f"{row['id']} missing {key}"
            if status != "unavailable":
                assert store.result(tool, ticker, period).status in {"ok", "stale"}


def test_gold_trajectories_end_with_finish():
    for row in examples():
        tools = row["gold_tools"]
        assert tools, row["id"]
        assert tools[-1] == "finish", row["id"]
        assert tools.count("finish") == 1


def test_abstention_examples_are_marked_and_present():
    rows = examples()
    abstaining = [row for row in rows if row["expect_abstain"]]
    assert abstaining
    assert any(row["intent"] == "advice_request" for row in abstaining)


def test_test_set_is_english_only():
    assert {row["language"] for row in examples()} == {"en"}


def test_conflict_intent_is_deferred_until_sec_exists():
    assert "conflict" not in {row["intent"] for row in examples()}
