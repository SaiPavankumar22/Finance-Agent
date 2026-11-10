"""Full QLoRA run on the 620 training rows, then evaluation on both held-out sets.

    modal run training/train_full.py

Protocol:
- Train: `data/splits/train.jsonl` (620 rows) only. The 51 validation rows and
  the 150 frozen test rows are asserted disjoint from it, and the validation
  tickers are asserted absent from training.
- Evaluate: the baseline's own system prompt and JSON parser are imported from
  `eval/run_baseline.py`, so the before/after numbers answer the same question
  as the WS3 baseline does.
- The frozen test set is checksummed before and after the run; nothing writes
  to it.

Metrics are the WS3 ones (`eval/metrics.py`): schema validity, tool selection,
argument accuracy, and abstention precision/recall. The base model is evaluated
locally with the adapter disabled, so base and tuned differ only by the adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal

MODEL_ID = "CohereLabs/tiny-aya-global"
GPU_TYPE = os.getenv("MODAL_GPU", "A100-40GB")
SRC = "/src"
TRAIN_PATH = "/data/train.jsonl"
VAL_PATH = "/data/val.jsonl"
TESTSET_PATH = f"{SRC}/eval/testset.jsonl"

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
        # eval.run_baseline imports app.tools.registry, whose package __init__
        # pulls the yfinance layer; both are needed only to build the prompt.
        "pandas",
        "yfinance",
    )
    .env(
        {
            # 6464-token examples against a 262k vocab leave the allocator
            # fragmented; the local export never reaches the container.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir("app", f"{SRC}/app", ignore=["__pycache__", "*.pyc"])
    .add_local_file("eval/__init__.py", f"{SRC}/eval/__init__.py")
    .add_local_file("eval/metrics.py", f"{SRC}/eval/metrics.py")
    .add_local_file("eval/run_baseline.py", f"{SRC}/eval/run_baseline.py")
    .add_local_file("eval/testset.jsonl", TESTSET_PATH)
    .add_local_file("training/__init__.py", f"{SRC}/training/__init__.py")
    .add_local_file("training/data.py", f"{SRC}/training/data.py")
    .add_local_file("data/splits/train.jsonl", TRAIN_PATH)
    .add_local_file("data/splits/val.jsonl", VAL_PATH)
)

app = modal.App("finance-agent-sft-full", image=image)
adapters = modal.Volume.from_name("finance-agent-sft-adapters", create_if_missing=True)


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_row(row: dict[str, Any], prediction: Any) -> dict[str, Any]:
    """Score one first-turn prediction against that row's gold call.

    Frozen-test rows carry ``gold_tools`` and are scored with the baseline's own
    derivation; split rows carry a full trajectory instead, so the gold call is
    parsed from their first assistant turn. The two paths are checked against
    each other on the validation rows, the only rows that have both shapes.
    """
    from eval.metrics import abstention_decision, expected_first_call
    from training.data import expects_abstention, gold_first_call

    if "messages" in row:
        gold_tool, gold_args = gold_first_call(row)
    else:
        gold_tool, gold_args = expected_first_call(row)
    expected = expects_abstention(row)
    decision = abstention_decision(prediction)
    declined = decision == "abstain"
    return {
        "id": row["id"],
        "intent": row["intent"],
        "expected_tool": gold_tool,
        "predicted_tool": prediction.tool,
        "schema_valid": prediction.schema_valid,
        "tool_correct": prediction.tool == gold_tool,
        "args_correct": prediction.args == gold_args,
        "abstention_expected": expected,
        "abstention_decision": decision,
        "abstention_signal": declined,
        "abstention_correct": declined == expected,
        "raw_text": prediction.raw_text,
    }


@app.function(gpu=GPU_TYPE, timeout=10800, volumes={"/out": adapters}, max_containers=1)
def run(
    epochs: int = 2,
    learning_rate: float = 1e-4,
    rank: int = 16,
    grad_accum: int = 16,
    max_len: int = 6464,
    max_new_tokens: int = 128,
    seed: int = 13,
    expected_testset_sha256: str = "",
) -> dict[str, Any]:
    import torch
    import transformers
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    sys.path.insert(0, SRC)
    from app.chat_format import render
    from eval.metrics import expected_first_call, summarize, wilson_interval
    from eval.run_baseline import SYSTEM_PROMPT, parse_prediction
    from training.data import (
        assert_disjoint,
        build_example,
        describe_masking,
        gold_first_call,
        load_jsonl,
    )

    started = time.time()
    train_rows = load_jsonl(TRAIN_PATH)
    val_rows = load_jsonl(VAL_PATH)
    test_rows = load_jsonl(TESTSET_PATH)
    data_checks = assert_disjoint(train_rows, val_rows, test_rows)

    digest_before = sha256(TESTSET_PATH)
    if expected_testset_sha256 and digest_before != expected_testset_sha256:
        raise AssertionError(f"frozen test set changed: {digest_before}")

    # Validation rows are scored from their trajectory, so the baseline's own
    # derivation must reproduce the same call before those numbers mean anything.
    agreement = sum(
        1
        for row in val_rows
        if expected_first_call({**row, "gold_tools": [gold_first_call(row)[0]]})
        == gold_first_call(row)
    )
    data_checks["val_gold_path_agreement"] = agreement
    data_checks["val_gold_path_disagreements"] = len(val_rows) - agreement
    if agreement != len(val_rows):
        raise AssertionError("baseline gold derivation disagrees with the trajectories")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    examples = [build_example(tokenizer, row["messages"], max_len) for row in train_rows]
    over_budget = sum(1 for example in examples if example["full_tokens"] > max_len)
    supervised_total = sum(example["supervised"] for example in examples)
    masking = describe_masking(tokenizer, examples[0])

    model = AutoModelForCausalLM.from_pretrained(
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
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.05,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    micro_steps = epochs * len(examples)
    steps_per_epoch = math.ceil(len(examples) / grad_accum)
    optimizer_steps = epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=optimizer_steps, pct_start=0.03
    )

    def generate(question: str) -> Any:
        prompt = render(
            tokenizer,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        text = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return parse_prediction(text)

    def evaluate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        try:
            scored = [score_row(row, generate(row["question"])) for row in rows]
        finally:
            # Gradient checkpointing must come back with PEFT's input-grad hook,
            # otherwise the checkpointed path is inactive and the next training
            # step keeps every activation: 73 GiB allocated, OOM at epoch 2.
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.config.use_cache = False
            model.train()
            torch.cuda.empty_cache()
        print(
            f"eval done; checkpointing back on, {torch.cuda.memory_allocated() / 1024**3:.1f} GiB"
            " allocated",
            flush=True,
        )
        return scored

    def accuracy(scored: list[dict[str, Any]]) -> float:
        return round(sum(row["tool_correct"] for row in scored) / len(scored), 4)

    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, Any]] = []
    epoch_val: list[dict[str, Any]] = []
    best: dict[str, Any] = {"accuracy": -1.0, "epoch": 0, "state": None}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    trained_at = time.time()

    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(examples), generator=generator).tolist()
        running: list[float] = []
        for position, index in enumerate(order, start=1):
            example = examples[index]
            input_ids = torch.tensor([example["input_ids"]], device="cuda")
            labels = torch.tensor([example["labels"]], device="cuda")
            out = model(input_ids=input_ids, labels=labels)
            (out.loss / grad_accum).backward()
            running.append(float(out.loss.detach()))
            finished = position % grad_accum == 0 or position == len(order)
            if finished:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
                history.append(
                    {
                        "epoch": epoch,
                        "micro_step": position,
                        "loss": round(sum(running) / len(running), 4),
                    }
                )
                running = []
                print(
                    f"epoch {epoch} step {len(history)}/{optimizer_steps} "
                    f"loss {history[-1]['loss']:.4f}",
                    flush=True,
                )
        val_scored = evaluate(val_rows)
        epoch_val.append(
            {
                "epoch": epoch,
                "val_tool_selection_accuracy": accuracy(val_scored),
                "val_argument_accuracy": round(
                    sum(row["args_correct"] for row in val_scored) / len(val_scored), 4
                ),
            }
        )
        print(f"epoch {epoch} val {epoch_val[-1]}", flush=True)
        if epoch_val[-1]["val_tool_selection_accuracy"] > best["accuracy"]:
            best = {
                "accuracy": epoch_val[-1]["val_tool_selection_accuracy"],
                "epoch": epoch,
                "state": {
                    key: value.detach().cpu().clone()
                    for key, value in get_peft_model_state_dict(model).items()
                },
            }
            # Persist the best epoch now: a later crash then costs a rerun of the
            # evaluation only, not the training.
            best_dir = f"/out/adapter-r{rank}-e{epochs}-lr{learning_rate}-best"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            adapters.commit()
            print(f"epoch {epoch} is best so far; adapter saved to {best_dir}", flush=True)
        model.train()

    train_seconds = time.time() - trained_at

    if best["state"] is not None:
        set_peft_model_state_dict(model, best["state"])

    tuned_val = evaluate(val_rows)
    tuned_test = evaluate(test_rows)
    model.disable_adapter_layers()
    base_val = evaluate(val_rows)
    base_test = evaluate(test_rows)
    model.enable_adapter_layers()

    def block(scored: list[dict[str, Any]]) -> dict[str, Any]:
        summary = summarize(scored)
        summary["tool_selection_95ci"] = wilson_interval(
            sum(row["tool_correct"] for row in scored), len(scored)
        )
        summary["argument_accuracy_95ci"] = wilson_interval(
            sum(row["args_correct"] for row in scored), len(scored)
        )
        return summary

    def paired(base: list[dict[str, Any]], tuned: list[dict[str, Any]]) -> dict[str, Any]:
        base_by_id = {row["id"]: row for row in base}
        counts = {"gained": 0, "lost": 0, "both": 0, "neither": 0}
        arg_counts = {"gained": 0, "lost": 0, "both": 0, "neither": 0}
        for row in tuned:
            other = base_by_id[row["id"]]
            for bucket, key in ((counts, "tool_correct"), (arg_counts, "args_correct")):
                if row[key] and other[key]:
                    bucket["both"] += 1
                elif row[key]:
                    bucket["gained"] += 1
                elif other[key]:
                    bucket["lost"] += 1
                else:
                    bucket["neither"] += 1
        return {"tool_selection": counts, "argument_accuracy": arg_counts}

    adapter_dir = f"/out/adapter-r{rank}-e{epochs}-lr{learning_rate}"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    adapters.commit()

    digest_after = sha256(TESTSET_PATH)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    payload = {
        "model": MODEL_ID,
        "adapter": adapter_dir,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "peft": __import__("peft").__version__,
        },
        "config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "rank": rank,
            "alpha": 2 * rank,
            "dropout": 0.05,
            "grad_accum": grad_accum,
            "max_len": max_len,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
        },
        "train": {
            "rows": len(train_rows),
            "micro_steps": micro_steps,
            "optimizer_steps": optimizer_steps,
            "tokens_seen": sum(example["tokens"] for _ in range(epochs) for example in examples),
            "rows_over_max_len": over_budget,
            "supervised_tokens": supervised_total,
            "trainable_parameters": trainable,
            "total_parameters": total_params,
            "trainable_fraction": round(trainable / total_params, 6),
            "loss_history": history,
            "epoch_val": epoch_val,
            "best_epoch": best["epoch"],
            "seconds": round(train_seconds, 1),
        },
        "masking": masking,
        "data_checks": data_checks,
        "testset_sha256": {"before": digest_before, "after": digest_after},
        "eval": {
            "val": {"base": block(base_val), "tuned": block(tuned_val)},
            "test": {"base": block(base_test), "tuned": block(tuned_test)},
            "paired": {
                "val": paired(base_val, tuned_val),
                "test": paired(base_test, tuned_test),
            },
        },
        "predictions": {
            "val": {"base": base_val, "tuned": tuned_val},
            "test": {"base": base_test, "tuned": tuned_test},
        },
        "peak_gpu_gb": round(peak, 2),
        "total_seconds": round(time.time() - started, 1),
    }
    # The return value only exists on the client, so a client that dies mid-run
    # loses an hour of training. Keep the payload on the volume too.
    result_path = Path("/out/full.json")
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    adapters.commit()
    print(f"results written to /out/full.json ({peak:.1f} GiB peak)", flush=True)
    return payload


@app.local_entrypoint()
def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(".env", override=False)
    token = os.getenv("HF_SECONDARY_TOKEN") or os.getenv("HF_PRIMARY_TOKEN")
    if not token:
        raise SystemExit("Set HF_SECONDARY_TOKEN or HF_PRIMARY_TOKEN in .env")

    expected = Path("eval/testset.sha256").read_text(encoding="utf-8").split()[0]
    result = run.with_options(
        secrets=[modal.Secret.from_dict({"HF_TOKEN": token})]
    ).remote(
        epochs=int(os.getenv("SFT_EPOCHS", "2")),
        learning_rate=float(os.getenv("SFT_LR", "1e-4")),
        max_new_tokens=int(os.getenv("SFT_MAX_NEW", "128")),
        expected_testset_sha256=expected,
    )

    output = Path("artifacts/sft")
    output.mkdir(parents=True, exist_ok=True)
    (output / "full.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'set':6} {'model':7} {'tool':>6} {'args':>6} {'schema':>7} {'abstain R':>9}")
    for split in ("val", "test"):
        for name in ("base", "tuned"):
            summary = result["eval"][split][name]
            recall = summary["abstention"]["recall"]
            print(
                f"{split:6} {name:7} {summary['tool_selection_accuracy']:>6.4f} "
                f"{summary['argument_accuracy']:>6.4f} {summary['schema_valid_rate']:>7.4f} "
                f"{'n/a' if recall is None else f'{recall:.2f}':>9}"
            )
    print(f"\nbest epoch {result['train']['best_epoch']} "
          f"of {result['config']['epochs']}; adapter {result['adapter']}")
    unchanged = result["testset_sha256"]["before"] == result["testset_sha256"]["after"]
    print(f"test set unchanged: {unchanged}")
    print("Saved artifacts/sft/full.json")
