"""Verify Tiny Aya Global's chat format and JSON-schema decoding on Modal.

Run from the repository root:

    modal run modal/verify_tiny_aya.py

The local process passes HF_SECONDARY_TOKEN to Modal as an ephemeral secret.
The token is never printed or written to the repository.
"""

from __future__ import annotations

import json
import os
from typing import Any

import modal

MODEL_ID = "CohereLabs/tiny-aya-global"
SYSTEM_PROMPT = (
    'Emit exactly one JSON object: '
    '{"tool":"get_stock_quote","args":{"ticker":"AAPL"}}'
)

# Based on Modal's official GPU example:
# https://modal.com/docs/examples/install_cuda
image = modal.Image.from_registry(
    "vllm/vllm-openai:latest",
    add_python="3.12",
).entrypoint([])
app = modal.App("finance-agent-tiny-aya-verification", image=image)

TOOL_CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["get_stock_quote"]},
        "args": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    "required": ["tool", "args"],
    "additionalProperties": False,
}


@app.function(
    gpu="T4",
    timeout=900,
    scaledown_window=60,
    max_containers=1,
)
def verify() -> dict[str, Any]:
    """Load the model once and test both template rendering and constrained JSON."""

    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Get the current quote for AAPL."},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    template_has_assistant_prompt = rendered.rstrip().endswith(
        (
            "<|START_RESPONSE|>",
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
            "<|assistant|>",
        )
    )

    sampling = SamplingParams(
        max_tokens=64,
        temperature=0,
        structured_outputs=StructuredOutputsParams(json=TOOL_CALL_SCHEMA),
    )
    output = llm.generate(rendered, sampling_params=sampling)[0]
    generated = output.outputs[0].text.strip()
    try:
        parsed = json.loads(generated)
        schema_valid = (
            isinstance(parsed, dict)
            and parsed.get("tool") == "get_stock_quote"
            and parsed.get("args", {}).get("ticker") == "AAPL"
        )
        parse_error = None
    except json.JSONDecodeError as error:
        parsed = None
        schema_valid = False
        parse_error = str(error)

    return {
        "model": MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "chat_template": tokenizer.chat_template is not None,
        "rendered_prompt": rendered,
        "template_has_assistant_prompt": template_has_assistant_prompt,
        "constrained_output": generated,
        "parsed_output": parsed,
        "schema_valid": schema_valid,
        "parse_error": parse_error,
        "decision": (
            "use native chat template and vLLM JSON schema decoding"
            if template_has_assistant_prompt and schema_valid
            else "use explicit prompt wrapper plus validate-and-retry fallback"
        ),
    }


@app.local_entrypoint()
def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(override=False)
    token = os.getenv("HF_SECONDARY_TOKEN") or os.getenv("HF_PRIMARY_TOKEN")
    if not token:
        raise SystemExit("Set HF_SECONDARY_TOKEN or HF_PRIMARY_TOKEN in .env")

    # The secret exists only for this Modal invocation and is not persisted here.
    result = verify.with_options(
        secrets=[modal.Secret.from_dict({"HF_TOKEN": token})]
    ).remote()
    print(json.dumps(result, indent=2))
