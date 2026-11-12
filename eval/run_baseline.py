"""Run WS3 first-turn baselines on the frozen test set.

Examples:
    python -m eval.run_baseline --models tiny-aya-global
    python -m eval.run_baseline --models tiny-aya-base,tiny-aya-global,anthropic
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eval.metrics import Prediction, score_prediction, summarize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTSET = PROJECT_ROOT / "eval" / "testset.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
CACHE_DIR = RESULTS_DIR / "cache"

#: The WS6 prompt is the single planner prompt: it carries the same tool block
#: as before plus the refusal affordance, and every row now trains on it, so
#: evaluation must send exactly this text or it measures a prompt shift.
from eval.safety import SAFETY_SYSTEM_PROMPT as SYSTEM_PROMPT  # noqa: E402


def load_rows(limit: int | None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in TESTSET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def parse_prediction(text: str) -> Prediction:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    candidates = [cleaned]
    candidates.extend(re.findall(r"\{.*\}", cleaned, flags=re.DOTALL))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if not isinstance(payload, dict):
                continue
            tool = payload.get("tool")
            args = payload.get("args", {})
            if isinstance(tool, str) and isinstance(args, dict):
                return Prediction(cleaned, tool, args, True)
        except json.JSONDecodeError:
            continue
    return Prediction(cleaned, None, {}, False)


def cache_key(model_name: str, row: dict[str, Any]) -> Path:
    digest = hashlib.sha256(
        f"ws3-v2:{model_name}:{row['id']}:{row['question']}".encode("utf-8")
    ).hexdigest()
    return CACHE_DIR / model_name.replace("/", "__") / f"{digest}.txt"


def call_huggingface(model: str, question: str) -> str:
    from huggingface_hub import InferenceClient

    token = os.environ["HF_SECONDARY_TOKEN"]
    client = InferenceClient(provider="cohere", model=model, api_key=token)
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=128,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def call_anthropic(question: str) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    response = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-haiku-5-5"),
        max_tokens=128,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return "".join(block.text for block in response.content if hasattr(block, "text"))


def run_model(model_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    scored: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        path = cache_key(model_name, row)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raw = path.read_text(encoding="utf-8")
        else:
            try:
                raw = (
                    call_anthropic(row["question"])
                    if model_name == "anthropic"
                    else call_huggingface(model_name, row["question"])
                )
            except Exception as error:
                return {
                    "model": model_name,
                    "status": "unavailable",
                    "error": f"{type(error).__name__}: {error}",
                    "summary": {"examples": 0},
                    "predictions": [],
                }
            path.write_text(raw, encoding="utf-8")
        scored.append(score_prediction(row, parse_prediction(raw)))
        if index % 10 == 0 or index == len(rows):
            print(f"{model_name}: {index}/{len(rows)}")
    return {
        "model": model_name,
        "summary": summarize(scored),
        "predictions": scored,
    }


def main() -> int:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="tiny-aya-global",
        help="Comma-separated: tiny-aya-base, tiny-aya-global, anthropic",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model_aliases = {
        "tiny-aya-base": "CohereLabs/tiny-aya-base",
        "tiny-aya-global": "CohereLabs/tiny-aya-global",
        "anthropic": "anthropic",
    }
    requested = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = set(requested) - set(model_aliases)
    if unknown:
        raise SystemExit(f"Unknown models: {', '.join(sorted(unknown))}")

    rows = load_rows(args.limit)
    results = [run_model(model_aliases[name], rows) for name in requested]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": "1.0",
        "testset": str(TESTSET.relative_to(PROJECT_ROOT)),
        "examples": len(rows),
        "models": results,
        "budget_note": "First-turn baseline; no live market tools are called.",
    }
    output_path = RESULTS_DIR / "baseline.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
