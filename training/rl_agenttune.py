"""GRPO pass on the planner task through AgentTune's agentic trainer.

    modal run training/rl_agenttune.py            # 8-step smoke run
    modal run training/rl_agenttune.py --full     # longer run

The RL stage asks the model for a tool call, runs the tool, and rewards the
episode. Tools here are deterministic stubs that return the shape of a recorded
`ToolResult`, because a rollout that hits Yahoo would be slow, rate-limited and
non-reproducible; the reward therefore measures the planning decision, which is
what the SFT stage was trained on.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import modal

MODEL_ID = "CohereLabs/tiny-aya-global"
GPU_TYPE = os.getenv("MODAL_GPU", "A100-40GB")
TRAIN_PATH = "/data/train.jsonl"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("torch", "datasets", "git+https://github.com/Lexsi-Labs/AgentTune.git")
    .env({"TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("app", "/src/app", ignore=["__pycache__", "*.pyc"])
    .add_local_file("data/splits/train.jsonl", TRAIN_PATH)
)

app = modal.App("finance-agent-agenttune-rl", image=image)
adapters = modal.Volume.from_name("finance-agent-sft-adapters", create_if_missing=True)


def _stub(tool: str, ticker: str, **extra: Any) -> dict[str, Any]:
    """A recorded-shaped ToolResult, so rollouts never touch the provider."""

    return {
        "status": "ok",
        "tool": tool,
        "args": {"ticker": ticker, **extra},
        "data": {"ticker": ticker, "price": 100.0, "currency": "USD", "observations": 240},
        "source": "fixture",
        "error": None,
    }


def get_stock_quote(ticker: str) -> dict[str, Any]:
    """Latest price for one exchange symbol, e.g. RELIANCE.NS or AAPL."""

    return _stub("get_stock_quote", ticker)


def get_financial_ratios(ticker: str) -> dict[str, Any]:
    """Valuation and leverage ratios for one exchange symbol."""

    return _stub("get_financial_ratios", ticker)


def get_historical_prices(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Daily closes and volatility for one symbol over one period."""

    return _stub("get_historical_prices", ticker, period=period)


def get_company_news(ticker: str, limit: int = 5) -> dict[str, Any]:
    """Recent headlines for one exchange symbol."""

    return _stub("get_company_news", ticker, limit=limit)


@app.function(
    gpu=GPU_TYPE,
    timeout=10800,
    volumes={"/out": adapters},
    max_containers=1,
    secrets=[modal.Secret.from_name("hf-read-token")],
)
def run(
    max_samples: int = 40,
    steps: int = 8,
    learning_rate: float = 1e-6,
    num_generations: int = 2,
    output: str = "agenttune-grpo-smoke",
) -> dict[str, Any]:
    import sys

    from datasets import Dataset
    from transformers import AutoTokenizer

    sys.path.insert(0, "/src")
    from app.chat_format import render

    started = time.time()
    rows = [
        json.loads(line)
        for line in Path(TRAIN_PATH).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:max_samples]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    records = []
    for row in rows:
        messages = row["messages"]
        final = max(i for i, message in enumerate(messages) if message["role"] == "assistant")
        prompt = render(tokenizer, messages[:final], add_generation_prompt=True)
        records.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "answer": messages[final]["content"],
            }
        )
    dataset = Dataset.from_list(records)
    print(f"rows: {len(dataset)} | sample answer: {records[0]['answer'][:80]!r}", flush=True)

    from agenttune.core.backend_factory import create_agentic_trainer

    trainer = create_agentic_trainer(
        algorithm="grpo",
        model=MODEL_ID,
        train_dataset=dataset,
        tools=[get_stock_quote, get_financial_ratios, get_historical_prices, get_company_news],
        reward_funcs=["structure_reward", "reward_tool_used"],
        output_dir=f"/out/{output}",
        max_steps=steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_generations=num_generations,
        report_to="none",
    )
    trainer.train()

    out_dir = Path(f"/out/{output}")
    adapters.commit()
    files = sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())
    return {
        "output_dir": str(out_dir),
        "files": files[:30],
        "rows": len(dataset),
        "steps": steps,
        "elapsed_s": round(time.time() - started),
    }


@app.local_entrypoint()
def main(full: bool = False, output: str = "agenttune-grpo-smoke") -> None:
    result = run.remote(steps=120 if full else 8, max_samples=400 if full else 40, output=output)
    print(json.dumps(result, indent=2, default=str)[:3000])
