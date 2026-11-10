"""QLoRA smoke run for Tiny Aya Global on the WS4 training split.

This is the plan's dry run (section 12, step 1): ~40 optimizer steps on a slice
of the 620 training rows, enough to show the loss moving and to prove that the
labels cover assistant tokens only. It is not the full run and it does not push
an adapter.

    SFT_STEPS=40 modal run training/train_qlora.py

Design notes:
- The tool-result mapping comes from ``app/chat_format.py`` so training and the
  future endpoint render trajectories identically; there is no second copy.
- Labels are built by rendering every prefix with ``add_generation_prompt=True``
  and asserting it is a token-prefix of the full rendering. That assertion is
  what makes "assistant-only" verifiable rather than assumed.
- 96 of the 620 rows are 11-turn trajectories with 250-point price histories
  (up to 6403 tokens), so the plan's 2048-4096 window is too small; the run
  uses 6144 with gradient checkpointing instead of silently truncating them.
- Hardware: a 22 GB A10G runs out of memory at 6144 tokens, because a 262k-entry
  vocabulary makes the logits tensor about 6 GB on its own. A 40 GB A100 peaks
  at 29.6 GB, so `MODAL_GPU=A100-40GB` is the setting that works. The 4-bit
  quantizer must also skip `lm_head`: quantizing a head that wide degenerates
  into repeated single tokens, which looked like a training bug until fixed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal

MODEL_ID = "CohereLabs/tiny-aya-global"
GPU_TYPE = os.getenv("MODAL_GPU", "A10G")
TRAIN_PATH = "/data/train.jsonl"
SRC_PATH = "/src"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
    )
    .add_local_file("data/splits/train.jsonl", TRAIN_PATH)
    .add_local_file("app/chat_format.py", f"{SRC_PATH}/app/chat_format.py")
)

app = modal.App("finance-agent-sft-smoke", image=image)
hf_cache = modal.Volume.from_name("finance-agent-hf-cache", create_if_missing=True)


def build_example(tokenizer: Any, messages: list[dict[str, Any]], max_len: int) -> dict[str, Any]:
    """Tokenize one trajectory and mask everything except assistant turns."""
    from app.chat_format import tokenize

    full = tokenize(tokenizer, messages)
    labels = [-100] * len(full)
    spans: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tokenize(tokenizer, messages[:index], add_generation_prompt=True)
        upto = tokenize(tokenizer, messages[: index + 1])
        if upto[: len(prefix)] != prefix:
            raise AssertionError(f"prefix not stable at turn {index}")
        labels[len(prefix) : len(upto)] = upto[len(prefix) :]
        spans.append((len(prefix), len(upto)))
    if len(full) != len(labels):
        raise AssertionError("label/input length mismatch")
    return {
        "input_ids": full[:max_len],
        "labels": labels[:max_len],
        "spans": spans,
        "full_tokens": len(full),
        "supervised": sum(1 for value in labels[:max_len] if value != -100),
        "tokens": min(len(full), max_len),
    }


def describe_masking(tokenizer: Any, example: dict[str, Any]) -> dict[str, Any]:
    """Prove the mask by decoding exactly what the loss will see."""
    mask = [value != -100 for value in example["labels"]]
    first = mask.index(True)
    run_end = first
    while run_end < len(mask) and mask[run_end]:
        run_end += 1
    supervised_ids = [token for token, keep in zip(example["input_ids"], mask, strict=True) if keep]
    return {
        "input_tokens": len(example["input_ids"]),
        "masked_tokens": mask.count(False),
        "supervised_tokens": len(supervised_ids),
        "first_supervised_index": first,
        "labels_are_neg100_before_first_assistant": all(
            value == -100 for value in example["labels"][:first]
        ),
        "decoded_prompt_tail": tokenizer.decode(example["input_ids"][:first][-24:]),
        "decoded_first_supervised_span": tokenizer.decode(example["input_ids"][first:run_end]),
        "all_supervision_is_assistant_text": all(
            any(start <= index < end for start, end in example["spans"])
            for index, keep in enumerate(mask)
            if keep
        ),
    }


@app.function(
    gpu=GPU_TYPE,
    timeout=3600,
    volumes={"/cache": hf_cache},
    max_containers=1,
)
def smoke(
    steps: int = 40,
    max_len: int = 6144,
    limit: int = 64,
    learning_rate: float = 2e-4,
    rank: int = 16,
    grad_accum: int = 4,
    seed: int = 13,
) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    sys.path.insert(0, SRC_PATH)
    from app.chat_format import render

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    rows = [
        json.loads(line)
        for line in Path(TRAIN_PATH).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:limit]

    examples = [build_example(tokenizer, row["messages"], max_len) for row in rows]
    over_budget = sum(1 for example in examples if example["full_tokens"] > max_len)
    supervised_total = sum(example["supervised"] for example in examples)
    print(f"built {len(examples)} examples, {over_budget} over {max_len} tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            # The 262k-entry vocabulary makes a 4-bit lm_head degenerate into
            # repeated single tokens; keep the head in bf16.
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
    total = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=steps, pct_start=0.1
    )

    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    tokens_seen = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    trained_at = time.time()
    for step in range(1, steps + 1):
        index = int(torch.randint(0, len(examples), (1,), generator=generator))
        example = examples[index]
        input_ids = torch.tensor([example["input_ids"]], device="cuda")
        labels = torch.tensor([example["labels"]], device="cuda")
        out = model(input_ids=input_ids, labels=labels)
        (out.loss / grad_accum).backward()
        if step % grad_accum == 0 or step == steps:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(round(float(out.loss.detach()), 4))
        tokens_seen += example["tokens"]
        if step % 5 == 0 or step == 1:
            print(f"step {step:3d} loss {losses[-1]:.4f} tokens {example['tokens']}", flush=True)
    train_seconds = time.time() - trained_at

    from app.chat_format import tokenize

    masking = describe_masking(tokenizer, examples[0])
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True

    def generate(prefix: list[dict[str, Any]], max_new_tokens: int = 48) -> dict[str, Any]:
        prompt = render(tokenizer, prefix, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        new = output[0][inputs["input_ids"].shape[1] :].tolist()
        return {
            "text": tokenizer.decode(new, skip_special_tokens=True).strip(),
            "new_tokens": len(new),
            "ended_turn": bool(new) and new[-1] == tokenizer.eos_token_id,
        }

    def next_token(prefix: list[dict[str, Any]], gold: str) -> dict[str, Any]:
        """Greedy next token at a given turn boundary, teacher-forced."""
        ids = tokenize(tokenizer, prefix, add_generation_prompt=True)
        with_ids = tokenize(tokenizer, prefix + [{"role": "assistant", "content": gold}])
        if with_ids[: len(ids)] != ids:
            raise AssertionError("prefix not stable for the probe")
        gold_first = with_ids[len(ids)]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1]
        probabilities = logits.softmax(-1)
        top = probabilities.topk(5)
        return {
            "gold_first_token": tokenizer.decode([gold_first], skip_special_tokens=True),
            "top1": tokenizer.decode([int(top.indices[0])], skip_special_tokens=True),
            "gold_in_top5": gold_first in top.indices.tolist(),
            "gold_probability": round(float(probabilities[gold_first]), 4),
        }

    first_row = rows[0]
    advice_row = next(row for row in rows if len(row["messages"]) == 3)
    probes = {
        "first_turn": (first_row, 2),
        "final_turn": (first_row, len(first_row["messages"]) - 1),
        "ood_advice": (advice_row, 2),
    }

    results: dict[str, Any] = {}
    for name, (row, upto) in probes.items():
        prefix = row["messages"][:upto]
        gold = row["messages"][upto]["content"]
        model.disable_adapter_layers()
        base = generate(prefix)
        model.enable_adapter_layers()
        tuned = generate(prefix)
        results[name] = {
            "question": row["messages"][1]["content"],
            "intent": row["intent"],
            "gold": gold,
            "base": base,
            "tuned": tuned,
            "greedy_next_token": next_token(prefix, gold),
        }

    peak = torch.cuda.max_memory_allocated() / 1024**3
    return {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "peft": peft.__version__,
        },
        "steps": steps,
        "examples": len(examples),
        "max_len": max_len,
        "rows_over_max_len": over_budget,
        "losses": losses,
        "loss_first_5": round(sum(losses[:5]) / 5, 4),
        "loss_last_5": round(sum(losses[-5:]) / 5, 4),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": round(trainable / total, 6),
        "tokens_seen": tokens_seen,
        "supervised_tokens_in_slice": supervised_total,
        "mean_supervised_tokens_per_row": round(supervised_total / max(1, len(examples)), 1),
        "masking": masking,
        "probes": results,
        "train_seconds": round(train_seconds, 1),
        "total_seconds": round(time.time() - started, 1),
        "peak_gpu_gb": round(peak, 2),
    }


@app.local_entrypoint()
def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(".env", override=False)
    token = os.getenv("HF_SECONDARY_TOKEN") or os.getenv("HF_PRIMARY_TOKEN")
    if not token:
        raise SystemExit("Set HF_SECONDARY_TOKEN or HF_PRIMARY_TOKEN in .env")

    result = smoke.with_options(secrets=[modal.Secret.from_dict({"HF_TOKEN": token})]).remote(
        steps=int(os.getenv("SFT_STEPS", "40")),
        max_len=int(os.getenv("SFT_MAX_LEN", "6144")),
        limit=int(os.getenv("SFT_LIMIT", "64")),
    )
    output = Path("artifacts/sft")
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "losses"}
    print(json.dumps(summary, indent=2))
    print(f"\nSaved artifacts/sft/smoke.json ({len(result['losses'])} steps)")
