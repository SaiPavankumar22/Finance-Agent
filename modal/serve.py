"""Public inference service for the hackathon demo.

Deploy with:
    modal deploy modal/serve.py

The service loads the accepted adapter from the Hub at a pinned revision, so the
demo runs the same weights the gate report was measured on rather than whatever
happens to sit on the volume. It exposes a small, auditable API: the model
selects one registered tool, the tool is executed, and both are returned.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

sys.path.insert(0, "/src")
from finance_agent.release import ADAPTER_REPO, ADAPTER_REVISION, MODEL_ID

ROOT = Path(__file__).resolve().parents[1]
SRC = "/src"
ADAPTER = f"{ADAPTER_REPO}@{ADAPTER_REVISION}"

#: One JSON object per request on stdout, so Modal's log stream is queryable by
#: request id and a bad answer can be traced to its question, planner action and
#: tool result.
logger = logging.getLogger("finance_agent.requests")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

#: Log lines are also mirrored into a Modal Dict, so the ``request_log`` endpoint
#: can return one line in the browser instead of making the reader find it in the
#: dashboard log stream. Entries older than the retention window are dropped.
LOG_STORE_NAME = "finance-agent-request-log"
LOG_RETENTION_SECONDS = 7 * 24 * 60 * 60
LOG_PRUNE_EVERY = 100

_LOG_STORE: modal.Dict | None = None
_LOG_WRITES = 0


def _log_store() -> modal.Dict | None:
    """Lazy Dict handle, or ``None`` when the store cannot be resolved."""

    global _LOG_STORE
    if _LOG_STORE is None:
        try:
            _LOG_STORE = modal.Dict.from_name(LOG_STORE_NAME, create_if_missing=True)
        except Exception as error:  # logging must never break a request
            logger.warning("request log store unavailable: %s", error)
            return None
    return _LOG_STORE


def _prune(store: modal.Dict) -> None:
    """Drop entries past the retention window (called rarely, never per request)."""

    cutoff = time.time() - LOG_RETENTION_SECONDS
    try:
        stale = [key for key, value in store.items() if _logged_at(value) < cutoff]
        for key in stale:
            store.pop(key, None)
    except Exception as error:
        logger.warning("request log prune skipped: %s", error)


def _logged_at(record: Any) -> float:
    try:
        stamp = datetime.strptime(str(record.get("ts")), "%Y-%m-%dT%H:%M:%SZ")
    except (AttributeError, TypeError, ValueError):
        return 0.0
    return stamp.replace(tzinfo=timezone.utc).timestamp()


def _mirror(record: dict[str, Any], request_id: str) -> None:
    store = _log_store()
    if store is None:
        return
    global _LOG_WRITES
    try:
        store[request_id] = record
        _LOG_WRITES += 1
        if _LOG_WRITES % LOG_PRUNE_EVERY == 0:
            _prune(store)
    except Exception as error:
        logger.warning("request log mirror failed: %s", error)


def log_request(**fields: Any) -> None:
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
    logger.info(json.dumps(record, default=str))
    request_id = record.get("request_id")
    if isinstance(request_id, str) and request_id and request_id != "-":
        _mirror(record, request_id)


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
    .add_local_dir("src/finance_agent", f"{SRC}/finance_agent")
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


@app.cls(
    gpu="A10G",
    timeout=300,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("hf-read-token")],
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
        self.model = PeftModel.from_pretrained(
            base, ADAPTER_REPO, revision=ADAPTER_REVISION, token=os.getenv("HF_TOKEN")
        )
        self.model.eval()

    @modal.method()
    def query_model(self, question: str, request_id: str = "-") -> dict[str, Any]:
        from training.train_full import generate_first_call

        started = time.perf_counter()
        prediction = generate_first_call(self.model, self.tokenizer, question)
        # ``generate_first_call`` returns a ``Prediction`` dataclass, not a dict,
        # so the served demo used to skip tool execution entirely and answer with
        # a null result. Normalise before looking at the chosen tool.
        if isinstance(prediction, dict):
            action = prediction
        elif is_dataclass(prediction):
            action = asdict(prediction)
        elif hasattr(prediction, "model_dump"):
            action = prediction.model_dump(mode="json")
        else:
            action = prediction
        if not isinstance(action, dict):
            log_request(
                request_id=request_id,
                event="planner",
                question=question,
                tool=None,
                args={},
                schema_valid=None,
                tool_status="unparsed",
                latency_ms=round((time.perf_counter() - started) * 1000),
                adapter=ADAPTER,
            )
            return {"action": action, "result": None}

        tool = action.get("tool")
        args = action.get("args", {})
        if tool == "finish":
            log_request(
                request_id=request_id,
                event="planner",
                question=question,
                tool=tool,
                args=args,
                schema_valid=action.get("schema_valid"),
                tool_status="declined",
                latency_ms=round((time.perf_counter() - started) * 1000),
                adapter=ADAPTER,
            )
            return {"action": action, "result": None}

        from app.tools import yfinance_tools

        functions = {
            "get_stock_quote": yfinance_tools.get_stock_quote,
            "get_financial_ratios": yfinance_tools.get_financial_ratios,
            "get_historical_prices": yfinance_tools.get_historical_prices,
            "get_company_news": yfinance_tools.get_company_news,
        }
        function = functions.get(tool)
        if function is None:
            log_request(
                request_id=request_id,
                event="planner",
                question=question,
                tool=tool,
                args=args,
                schema_valid=action.get("schema_valid"),
                tool_status="unknown_tool",
                latency_ms=round((time.perf_counter() - started) * 1000),
                adapter=ADAPTER,
            )
            return {"action": action, "result": {"error": "unknown_tool"}}
        try:
            result = function(**args)
        except Exception as exc:  # tool contract must not crash the endpoint
            result = {"error": "tool_failure", "detail": str(exc)}
        if result is None:
            result = {"status": "unavailable", "error": "provider_no_result"}
        if hasattr(result, "model_dump"):
            result = result.model_dump(mode="json")
        log_request(
            request_id=request_id,
            event="planner",
            question=question,
            tool=tool,
            args=args,
            schema_valid=action.get("schema_valid"),
            tool_status=(result or {}).get("status") if isinstance(result, dict) else None,
            tool_error=(result or {}).get("error") if isinstance(result, dict) else None,
            tool_error_code=(result or {}).get("error_code") if isinstance(result, dict) else None,
            latency_ms=round((time.perf_counter() - started) * 1000),
            adapter=ADAPTER,
        )
        return {"action": action, "result": result}

    @modal.fastapi_endpoint(method="POST", docs=True)
    def query(self, payload: dict[str, Any]) -> dict[str, Any]:
        from fastapi import HTTPException

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise HTTPException(status_code=400, detail="question must be a non-empty string")
        if len(question) > 1000:
            raise HTTPException(status_code=413, detail="question is too long")
        # The id is generated here, before the model runs, so a failed request
        # still gets an id the caller can quote back.
        request_id = uuid.uuid4().hex[:12]
        payload_out = self.query_model.local(question.strip(), request_id)
        return {
            "request_id": request_id,
            "model": MODEL_ID,
            "adapter": ADAPTER,
            "question": question,
            **payload_out,
        }

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    @modal.fastapi_endpoint(method="GET")
    def version(self) -> dict[str, str]:
        return {
            "model": MODEL_ID,
            "adapter": ADAPTER,
            "code": os.getenv("FINANCE_AGENT_SHA", "local"),
        }


#: The log lookup needs nothing but FastAPI, so it runs in a tiny CPU image
#: instead of waking (and paying for) the GPU container that serves answers. It
#: still needs ``finance_agent`` because this module imports the pinned release
#: metadata at import time.
log_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi")
    .add_local_dir("src/finance_agent", f"{SRC}/finance_agent")
)


@app.function(image=log_image, timeout=30)
@modal.fastapi_endpoint(method="GET", docs=True)
def request_log(request_id: str) -> dict[str, Any]:
    """Return the logged line for one request: ``?request_id=<id>``.

    Open it in a browser with the id the demo prints to trace an answer back to
    the question, the planner action and the tool result.
    """

    from fastapi import HTTPException

    store = _log_store()
    if store is None:
        raise HTTPException(status_code=503, detail="request log store unavailable")
    try:
        record = store.get(request_id)
    except Exception as error:  # noqa: BLE001 - surfaced as a 503, not a crash
        raise HTTPException(
            status_code=503, detail=f"request log store failed: {error}"
        ) from error
    if isinstance(record, dict):
        return record
    raise HTTPException(status_code=404, detail=f"no log line for request_id {request_id}")
