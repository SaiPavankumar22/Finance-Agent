"""The ToolResult envelope must make failure states impossible to misread."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.tools import ToolCall, ToolResult

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


def make(**overrides):
    payload = {
        "status": "ok",
        "tool": "get_stock_quote",
        "args": {"ticker": "AAPL"},
        "data": {"price": 189.23, "currency": "USD"},
        "source": "yfinance",
        "retrieved_at": NOW,
    }
    payload.update(overrides)
    return ToolResult(**payload)


def test_ok_result_requires_data():
    with pytest.raises(ValidationError):
        make(data=None)


def test_error_result_requires_error_code():
    with pytest.raises(ValidationError):
        make(status="error", data=None)


def test_error_result_is_not_usable():
    result = make(status="error", data=None, error="invalid_ticker")
    assert result.usable is False
    assert result.status == "error"


def test_stale_result_is_usable():
    result = make(status="stale", as_of=NOW)
    assert result.usable is True


def test_unavailable_result_needs_no_payload():
    result = make(status="unavailable", data=None)
    assert result.usable is False
    assert result.error is None


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValidationError):
        make(retrieved_at=datetime(2026, 11, 9, 12, 0))


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        make(confidence=0.9)


def test_tool_call_has_empty_args_by_default():
    call = ToolCall(tool="finish")
    assert call.args == {}
