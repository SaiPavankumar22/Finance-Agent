"""Render agent trajectories in Tiny Aya's chat format.

Tiny Aya Global ships a template that requires strict ``user``/``assistant``
alternation and has no ``tool`` role: passing ``role="tool"`` raises
``TemplateError: Conversation roles must alternate user/assistant/...``. Plan
section 10 fixes the consequence: tool results travel inside a user turn as

    [TOOL_RESULT call_id=c1] {"status": "ok", ...} [/TOOL_RESULT]

and training and inference must use this exact rendering, or the adapter learns
a format the endpoint never sends. Keeping the mapping in one module is the
cheap way to guarantee that; there is no second copy of this rule.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Sequence

TOOL_RESULT_TEMPLATE = "[TOOL_RESULT call_id={call_id}] {payload} [/TOOL_RESULT]"
CHAT_ROLES = ("system", "user", "assistant")


def tool_result_text(content: str) -> str:
    """Wrap a serialized tool result in the user-turn envelope.

    The payload is re-serialized with sorted keys and no spaces so the same
    result always produces the same tokens, which keeps replay deterministic.
    """
    try:
        envelope = json.loads(content)
    except json.JSONDecodeError:
        return TOOL_RESULT_TEMPLATE.format(call_id="c?", payload=content.strip())
    call_id = str(envelope.get("call_id", "c?"))
    result = envelope.get("result", envelope)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return TOOL_RESULT_TEMPLATE.format(call_id=call_id, payload=payload)


def to_chat_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Map a trajectory onto alternating user/assistant turns.

    ``tool`` messages become user turns, which also restores the alternation a
    trajectory needs: system, user, assistant, [tool], assistant, [tool]...
    """
    converted: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "tool":
            converted.append({"role": "user", "content": tool_result_text(content)})
        elif role in CHAT_ROLES:
            converted.append({"role": role, "content": content})
        else:
            raise ValueError(f"unsupported role in trajectory: {role!r}")
    return converted


def check_alternation(messages: Sequence[dict[str, str]]) -> None:
    """Raise if the template would reject the conversation."""
    previous = "system"
    for message in messages:
        role = message["role"]
        if role == "system":
            if previous != "system":
                raise ValueError("system message outside the opening position")
            continue
        if previous == "system":
            if role != "user":
                raise ValueError(f"conversation opens with {role!r}, expected 'user'")
        elif role == previous:
            raise ValueError(f"consecutive {role!r} turns")
        previous = role


def render(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render a trajectory to text using the model's own chat template."""
    chat = to_chat_messages(messages)
    check_alternation(chat)
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def _flatten_ids(value: Any) -> list[int]:
    """Normalize whatever transformers returns into a flat id list.

    In transformers 5 ``apply_chat_template(tokenize=True)`` hands back a
    ``BatchEncoding`` for a single conversation, and older versions return a
    plain list, so both shapes are accepted here instead of at every call site.
    """
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if len(value) and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(token) for token in value]


def tokenize(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
) -> list[int]:
    """Tokenize a trajectory with the same mapping as :func:`render`."""
    chat = to_chat_messages(messages)
    check_alternation(chat)
    out = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    return _flatten_ids(out)
