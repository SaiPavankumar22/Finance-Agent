"""Run the SFT stage through AlignTune instead of the hand-written trainer.

    modal run training/train_aligntune.py            # 20-step chain check
    modal run training/train_aligntune.py --full     # 4-epoch run

AlignTune owns trainer construction, the 4-bit/QLoRA setup and checkpointing;
this module only supplies the project's data and settings, so the training half
of the LexiLabs chain is the library rather than a lookalike. AlignTune writes a
``lexsi_provenance.json`` next to the adapter, which is the file AuditKIT expects
when it audits a trained run.

The default is a 20-step run: it proves AlignTune can train this dataset on this
base model without spending a full GPU session, and it writes to its own output
directory so the published ``r32-e4`` adapter is never touched.
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
GPU_TYPE = os.getenv("MODAL_GPU", "A100-40GB")
SRC = "/src"
TRAIN_PATH = "/data/train.jsonl"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "aligntune", "datasets", "huggingface_hub")
    .env(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir("app", f"{SRC}/app", ignore=["__pycache__", "*.pyc"])
    .add_local_file("data/splits/train.jsonl", TRAIN_PATH)
)

app = modal.App("finance-agent-aligntune", image=image)
adapters = modal.Volume.from_name("finance-agent-sft-adapters", create_if_missing=True)


@app.function(
    gpu=GPU_TYPE,
    timeout=10800,
    volumes={"/out": adapters},
    max_containers=1,
    secrets=[modal.Secret.from_name("hf-read-token")],
)
def run(
    epochs: int = 4,
    learning_rate: float = 1e-4,
    rank: int = 32,
    max_len: int = 6464,
    max_steps: int = 20,
    output: str = "aligntune-e4-r32",
    push_to: str = "",
) -> dict[str, Any]:
    from datasets import Dataset
    from transformers import AutoTokenizer

    started = time.time()
    rows = [
        json.loads(line)
        for line in Path(TRAIN_PATH).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Prompt/completion pairs, not raw conversations: TRL masks the completion
    # with ``completion_only_loss``, which reproduces this project's
    # assistant-only loss without depending on generation markers the Tiny Aya
    # chat template does not carry. One pair per assistant turn, so every turn of
    # a trajectory is supervised - the first library run only supervised the last
    # turn and lost the turn-after-tool-result behaviour.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    sys.path.insert(0, SRC)
    from app.chat_format import render

    def turn_pairs(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        """One prompt/completion pair per assistant turn, rendered identically."""
        pairs: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prompt = render(tokenizer, messages[:index], add_generation_prompt=True)
            through = render(tokenizer, messages[: index + 1])
            if not through.startswith(prompt):
                raise SystemExit(f"rendered prompt is not a prefix at turn {index}")
            pairs.append({"prompt": prompt, "completion": through[len(prompt) :]})
        return pairs

    records: list[dict[str, str]] = []
    dropped = 0
    for row in rows:
        for pair in turn_pairs(row["messages"]):
            tokens = len(tokenizer(pair["prompt"] + pair["completion"])["input_ids"])
            if tokens > max_len:
                dropped += 1
                continue
            # Same policy as training.data.expand_examples: repetition instead of
            # a weighted sampler, so steps and tokens stay explicit in the record.
            records.extend([pair] * int(row.get("weight", 1)))

    dataset = Dataset.from_list(records)
    print(
        f"rows: {len(rows)} -> samples: {len(dataset)} "
        f"(one per assistant turn, weighted; dropped over {max_len} tokens: {dropped})",
        flush=True,
    )
    print(f"first prompt tail: {dataset[0]['prompt'][-160:]!r}", flush=True)

    import aligntune
    from aligntune.core.backend_factory import create_sft_trainer

    print(f"aligntune {getattr(aligntune, '__version__', '?')}", flush=True)

    trainer = create_sft_trainer(
        model_name=MODEL_ID,
        dataset_name=dataset,
        backend="trl",
        output_dir=f"/out/{output}",
        num_epochs=epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        max_seq_length=max_len,
        batch_size=1,
        gradient_accumulation_steps=16,
        seed=13,
        use_lora=True,
        lora_r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.05,
        lora_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        quantization={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        },
        completion_only_loss=True,
        packing=False,
        push_to_hub=bool(push_to),
        hub_model_id=push_to or None,
    )
    history = trainer.train()

    out_dir = Path(f"/out/{output}")
    files = sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file())
    provenance = out_dir / "lexsi_provenance.json"
    adapters.commit()

    return {
        "output_dir": str(out_dir),
        "files": files[:40],
        "provenance": (
            json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else None
        ),
        "history": history if isinstance(history, dict) else str(history),
        "elapsed_s": round(time.time() - started),
        "max_steps": max_steps,
        "epochs": epochs,
    }


@app.local_entrypoint()
def main(
    full: bool = False,
    output: str = "aligntune-e4-r32",
    push_to: str = "",
    spawn: bool = False,
) -> None:
    """Train, or schedule the run and exit.

    ``--spawn`` fires the job and returns immediately: the call belongs to Modal
    rather than to this shell, so a dropped connection cannot cancel a two-hour
    training run (which is exactly what happened once).
    """

    kwargs = dict(
        max_steps=-1 if full else 20,
        output=output,
        push_to=push_to,
    )
    if spawn:
        call = run.spawn(**kwargs)
        print(f"spawned {call.object_id}")
        return
    result = run.remote(**kwargs)
    print(json.dumps(result, indent=2, default=str)[:4000])
