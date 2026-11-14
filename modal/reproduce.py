"""Re-score a slice from the Hub-published adapter, in a fresh container.

The accepted numbers were measured against the adapter directory on the Modal
volume. This loads the published copy from the Hub at a pinned revision instead,
so a wrong upload or a different base revision cannot hide behind a green gate
report.

    modal run modal/reproduce.py::main

Writes ``artifacts/sft/reproduce-<revision>.json``; :mod:`scripts.check_reproduce`
compares it with the accepted artifacts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal

sys.path.insert(0, "/src")
from finance_agent.release import ADAPTER_REPO, ADAPTER_REVISION, MODEL_ID

ROOT = Path(__file__).resolve().parents[1]
SRC = "/src"
FROZEN_SLICE = 12
SAFETY_SLICE = 10

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "pydantic",
        "python-dotenv",
        "pandas",
        "yfinance",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src/finance_agent", f"{SRC}/finance_agent")
    .add_local_dir("app", f"{SRC}/app", ignore=["__pycache__", "*.pyc"])
    .add_local_dir("eval", f"{SRC}/eval", ignore=["__pycache__", "*.pyc", "results"])
    .add_local_dir("training", f"{SRC}/training", ignore=["__pycache__", "*.pyc"])
)

app = modal.App("finance-agent-reproduce", image=image)


def _spread(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Take rows from every category before taking a second from any of them."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row["category"], []).append(row)
    picked: list[dict[str, Any]] = []
    depth = 0
    while len(picked) < limit:
        added = False
        for category in sorted(buckets):
            if depth < len(buckets[category]) and len(picked) < limit:
                picked.append(buckets[category][depth])
                added = True
        if not added:
            break
        depth += 1
    return picked


@app.function(timeout=300, secrets=[modal.Secret.from_name("hf-read-token")])
def hub_access() -> dict[str, str]:
    """Report which account the service token belongs to and whether it can read."""
    from huggingface_hub import HfApi, hf_hub_download

    token = os.getenv("HF_TOKEN")
    if not token:
        return {"token": "missing"}
    try:
        account = HfApi(token=token).whoami()["name"]
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        return {"token": "invalid", "error": type(exc).__name__}
    try:
        hf_hub_download(ADAPTER_REPO, "adapter_config.json", revision=ADAPTER_REVISION, token=token)
        readable = "yes"
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        readable = f"no ({type(exc).__name__})"
    return {"account": account, "repo": ADAPTER_REPO, "readable": readable}


@app.local_entrypoint()
def access() -> None:
    print(json.dumps(hub_access.remote(), indent=2))


@app.function(gpu="A10G", timeout=1800, secrets=[modal.Secret.from_name("hf-read-token")])
def reproduce(
    repo: str = ADAPTER_REPO,
    revision: str = ADAPTER_REVISION,
    frozen_slice: int = FROZEN_SLICE,
    safety_slice: int = SAFETY_SLICE,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    sys.path.insert(0, SRC)
    from eval.safety import load_rows, score_turn_one, turn_one_messages
    from training.data import load_jsonl
    from training.train_full import evaluate_rows, generate_action

    started = time.time()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set; the adapter repo is private")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head"],
        ),
    )
    model = PeftModel.from_pretrained(base, repo, revision=revision, token=token)
    model.eval()

    frozen_rows = load_jsonl(f"{SRC}/eval/testset.jsonl")[:frozen_slice]
    frozen = evaluate_rows(model, tokenizer, frozen_rows, 128, False)

    safety_rows = _spread(load_rows(f"{SRC}/eval/safety_set.jsonl"), safety_slice)
    safety = [
        score_turn_one(row, generate_action(model, tokenizer, turn_one_messages(row), 96))
        for row in safety_rows
    ]

    def rate(items: list[dict[str, Any]], key: str) -> float:
        return round(sum(bool(item[key]) for item in items) / len(items), 4)

    return {
        "model": MODEL_ID,
        "adapter": f"{repo}@{revision}",
        "gpu": torch.cuda.get_device_name(0),
        "decode": "greedy, stop at first complete JSON object",
        "frozen": {
            "rows": len(frozen),
            "tool_selection_accuracy": rate(frozen, "tool_correct"),
            "argument_accuracy": rate(frozen, "args_correct"),
            "schema_valid_rate": rate(frozen, "schema_valid"),
            "predictions": frozen,
        },
        "safety": {
            "rows": len(safety),
            "refusal_rows": sum(1 for item in safety if item["expect_refusal"]),
            "declined": sum(1 for item in safety if item["declined"]),
            "refusal_recall": round(
                sum(1 for item in safety if item["declined"])
                / max(1, sum(1 for item in safety if item["expect_refusal"])),
                4,
            ),
            "over_refusal": round(
                sum(1 for item in safety if item["declined"] and not item["expect_refusal"])
                / max(1, sum(1 for item in safety if not item["expect_refusal"])),
                4,
            ),
            "unsafe_text_rows": sum(1 for item in safety if item.get("unsafe_text")),
            "predictions": safety,
        },
        "total_seconds": round(time.time() - started, 1),
    }


@app.local_entrypoint()
def main() -> None:
    result = reproduce.remote()
    out = Path("artifacts/sft") / f"reproduce-{ADAPTER_REVISION}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "adapter": result["adapter"],
                "frozen": {
                    key: value for key, value in result["frozen"].items() if key != "predictions"
                },
                "safety": {
                    key: value for key, value in result["safety"].items() if key != "predictions"
                },
                "written": str(out),
            },
            indent=2,
        )
    )
