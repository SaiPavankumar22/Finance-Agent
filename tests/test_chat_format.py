import pytest

from app.chat_format import (
    TOOL_RESULT_TEMPLATE,
    check_alternation,
    to_chat_messages,
    tool_result_text,
)


def trajectory() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "tools"},
        {"role": "user", "content": "Tell me AAPL's P/E."},
        {
            "role": "assistant",
            "content": '{"tool":"get_financial_ratios","args":{"ticker":"AAPL"}}',
        },
        {
            "role": "tool",
            "content": '{"call_id": "c1", "result": {"status": "ok", "data": {"z": 1, "a": 2}}}',
        },
        {"role": "assistant", "content": '{"tool":"finish","args":{}}'},
    ]


def test_tool_results_become_user_turns():
    converted = to_chat_messages(trajectory())
    assert [message["role"] for message in converted] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert converted[3]["content"].startswith("[TOOL_RESULT call_id=c1] ")
    assert converted[3]["content"].endswith(" [/TOOL_RESULT]")


def test_tool_payload_is_serialized_deterministically():
    content = '{"call_id": "c2", "result": {"b": 2, "a": 1}}'
    assert tool_result_text(content) == TOOL_RESULT_TEMPLATE.format(
        call_id="c2", payload='{"a":1,"b":2}'
    )
    assert tool_result_text(content) == tool_result_text(content)


def test_unparseable_tool_content_is_still_wrapped():
    assert tool_result_text("boom") == TOOL_RESULT_TEMPLATE.format(call_id="c?", payload="boom")


def test_alternation_is_enforced():
    with pytest.raises(ValueError, match="consecutive"):
        check_alternation(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
            ]
        )
    check_alternation(to_chat_messages(trajectory()))


def test_unknown_roles_are_rejected():
    with pytest.raises(ValueError, match="unsupported role"):
        to_chat_messages([{"role": "developer", "content": "x"}])
