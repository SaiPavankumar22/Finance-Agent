"""Shared data helpers for the fine-tuning jobs.

Kept separate from the Modal apps so the training and evaluation code agree on
tokenization, gold extraction and abstention labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Behaviours whose gold trajectory withholds an answer.
ABSTAIN_BEHAVIORS = frozenset(
    {
        "refuse_advice",
        "abstain_invalid_ticker",
        "abstain_malformed_response",
        "stop_after_bounded_attempts",
    }
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    mask = [value != -100 for value in example["labels"]]
    first = mask.index(True)
    run_end = first
    while run_end < len(mask) and mask[run_end]:
        run_end += 1
    return {
        "input_tokens": len(example["input_ids"]),
        "masked_tokens": mask.count(False),
        "supervised_tokens": mask.count(True),
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


def gold_first_call(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The gold first action, read from the trajectory itself.

    Val rows come from the split and have no ``gold_tools`` field, so the gold
    call is parsed from the first assistant turn. On the frozen test set this is
    cross-checked against ``expected_first_call`` to prove both paths agree.
    """
    payload = json.loads(row["messages"][2]["content"])
    return payload["tool"], payload.get("args", {})


def expects_abstention(row: dict[str, Any]) -> bool:
    if "expect_abstain" in row:
        return bool(row["expect_abstain"])
    return row["expected_behavior"] in ABSTAIN_BEHAVIORS


def assert_disjoint(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fail loudly if validation or test data leaked into training."""
    train_questions = {row["question"] for row in train}
    train_tickers = {ticker for row in train for ticker in row["tickers"]}
    report = {
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_questions": len(train_questions),
        "val_questions_in_train": len({row["question"] for row in val} & train_questions),
        "test_questions_in_train": len({row["question"] for row in test} & train_questions),
        "val_tickers_in_train": sorted(
            {ticker for row in val for ticker in row["tickers"]} & train_tickers
        ),
        "train_unique_questions": len(train_questions) == len(train),
    }
    if report["val_questions_in_train"] or report["test_questions_in_train"]:
        raise AssertionError(f"question leakage into training: {report}")
    if report["val_tickers_in_train"]:
        raise AssertionError(f"validation tickers seen in training: {report}")
    return report
