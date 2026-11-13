"""Public WS5 inference service for the hackathon demo.

Deploy with:
    modal deploy modal/serve.py

The service deliberately uses the approved ``adapter-ws5`` volume entry, not
the rejected e4 checkpoints.  It exposes a small, auditable API: the model
selects one registered tool, the tool is executed, and both are returned.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
SRC = "/src"
MODEL_ID = "CohereLabs/tiny-aya-global"
# WS5 is stored under a wrapper directory in the shared volume.
ADAPTER_DIR = "/out/adapter-ws5/adapter-r16-e2-lr0.0001-best"

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
        "fastapi",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("app", f"{SRC}/app", ignore=["__pycache__", "*.pyc"])
    .add_local_file("eval/__init__.py", f"{SRC}/eval/__init__.py")
    .add_local_file("eval/metrics.py", f"{SRC}/eval/metrics.py")
    .add_local_file("eval/safety.py", f"{SRC}/eval/safety.py")
    .add_local_file("eval/run_baseline.py", f"{SRC}/eval/run_baseline.py")
    .add_local_file("training/__init__.py", f"{SRC}/training/__init__.py")
    .add_local_file("training/data.py", f"{SRC}/training/data.py")
    .add_local_file("training/train_full.py", f"{SRC}/training/train_full.py")
)

app = modal.App("finance-agent-ws5-demo", image=image)
adapters = modal.Volume.from_name("finance-agent-sft-adapters", create_if_missing=False)


@app.cls(
    gpu="A10G",
    timeout=300,
    scaledown_window=300,
    volumes={"/out": adapters},
    secrets=[modal.Secret.from_name("hf-token")],
    min_containers=0,
)
class FinanceAgent:
    @modal.enter()
    def load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        sys.path.insert(0, SRC)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
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
        self.model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        self.model.eval()

    @modal.method()
    def query_model(self, question: str) -> dict[str, Any]:
        from training.train_full import generate_first_call

        prediction = generate_first_call(self.model, self.tokenizer, question)
        if not isinstance(prediction, dict):
            return {"action": prediction, "result": None}

        tool = prediction.get("tool")
        args = prediction.get("args", {})
        if tool == "finish":
            return {"action": prediction, "result": None}

        from app.tools import yfinance_tools

        functions = {
            "get_stock_quote": yfinance_tools.get_stock_quote,
            "get_financial_ratios": yfinance_tools.get_financial_ratios,
            "get_historical_prices": yfinance_tools.get_historical_prices,
            "get_company_news": yfinance_tools.get_company_news,
        }
        function = functions.get(tool)
        if function is None:
            return {"action": prediction, "result": {"error": "unknown_tool"}}
        try:
            result = function(**args)
        except Exception as exc:  # tool contract must not crash the endpoint
            result = {"error": "tool_failure", "detail": str(exc)}
        if result is None:
            result = {"status": "unavailable", "error": "provider_no_result"}
        if hasattr(result, "model_dump"):
            result = result.model_dump(mode="json")
        return {"action": prediction, "result": result}

    @modal.fastapi_endpoint(method="POST", docs=True)
    def query(self, payload: dict[str, Any]) -> dict[str, Any]:
        from fastapi import HTTPException

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise HTTPException(status_code=400, detail="question must be a non-empty string")
        if len(question) > 1000:
            raise HTTPException(status_code=413, detail="question is too long")
        return {
            "model": MODEL_ID,
            "adapter": "SaiPavankumar22/finance-agent@7a4b00e",
            "question": question,
            **self.query_model.local(question.strip()),
        }

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    @modal.fastapi_endpoint(method="GET")
    def version(self) -> dict[str, str]:
        return {
            "model": MODEL_ID,
            "adapter": "SaiPavankumar22/finance-agent@7a4b00e",
            "code": os.getenv("FINANCE_AGENT_SHA", "local"),
        }
