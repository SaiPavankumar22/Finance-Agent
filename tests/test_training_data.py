"""Unit tests for the shared training/evaluation data helpers."""

from __future__ import annotations

import pytest

from training.data import first_json_object_complete


@pytest.mark.parametrize(
    ("text", "complete"),
    [
        ('{"tool": "get_stock_quote", "args": {"ticker": "DIS"}}', True),
        ('{"tool": "get_stock_quote", "args": {"ticker": "DIS"', False),
        ('{"tool": "get_historical_prices", "args": {"ticker": "PEP", "period', False),
        ('{"tool":"get_stock_quote","args":{"ticker":"DIS"}}{"tool":"get_', True),
        ('{"tool": "x", "args": {"note": "a } b"}}', True),
        ('{"tool": "x", "args": {"note": "a \\" } b"}}', True),
        ("I cannot give investment advice, but I can look up facts.", False),
        ("", False),
        ("{}", True),
        ("}{", False),
    ],
)
def test_first_json_object_complete(text: str, complete: bool) -> None:
    assert first_json_object_complete(text) is complete
