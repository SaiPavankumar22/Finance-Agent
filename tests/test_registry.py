"""The tool surface stays small and the terminal action stays explicit."""

from app.tools import TOOL_FUNCTIONS
from app.tools.registry import (
    FINISH_TOOL,
    TOOL_SPECS,
    evidence_fields,
    is_known_tool,
    tool_prompt,
)


def test_surface_is_four_tools_plus_finish():
    assert len(TOOL_SPECS) == 5
    assert set(TOOL_FUNCTIONS) == set(TOOL_SPECS) - {FINISH_TOOL}
    assert is_known_tool(FINISH_TOOL) is True
    assert is_known_tool("get_insider_trades") is False


def test_finish_is_terminal():
    assert TOOL_SPECS[FINISH_TOOL].terminal is True


def test_news_headlines_become_citable_evidence():
    fields = evidence_fields(
        "get_company_news", {"count": 2, "items": [{"title": "A"}, {"title": "B"}]}
    )
    labels = [field.resolved_label for field in fields]
    assert labels == ["headline_count", "headline_1", "headline_2"]


def test_prompt_lists_every_tool_and_the_json_contract():
    prompt = tool_prompt()
    for name in TOOL_SPECS:
        assert name in prompt
    assert '"tool"' in prompt
