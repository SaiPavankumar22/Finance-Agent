"""Compatibility-friendly runtime facade.

The original ``app`` package remains the stable implementation path for
existing pipelines and tests. New code should import the public contracts from
this facade so a future package move does not require another repository-wide
rewrite.
"""

from app.chat_format import check_alternation, render, tokenize
from app.tools import TOOL_FUNCTIONS
from app.tools.registry import TOOL_SPECS, tool_prompt

__all__ = [
    "TOOL_FUNCTIONS",
    "TOOL_SPECS",
    "check_alternation",
    "render",
    "tokenize",
    "tool_prompt",
]
