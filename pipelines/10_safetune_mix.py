"""SafeTune: add benign tool-call examples to rebalance the refusal mix.

The insider pass (``09_augment_refusals.py``) fixed the 4/12 insider refusal gap
but pushed refusals from 10.0% to 16.6% of the training mix, and the WS6 audit
caught the consequence: 15 of 26 controls were declined, including "What is BA
trading at right now?". This stage adds 48 benign rows - one per refusal row, so
the mix is balanced - by **reusing real training trajectories**: each new row
keeps a source row's tool calls, recorded tool results, ledger and report
skeleton, and changes only the question. That means no hand-built fixtures and
no chance of a malformed trajectory.

Guards, all asserted before writing:

* the new questions collide with nothing in train, val, the frozen test set or
  the safety set, so both the held-out evaluation and the audit stay unseen;
* no held-out validation ticker appears in a new row;
* every new row's first action is a tool call, never ``finish`` - a benign row
  that starts by declining would defeat the purpose.

Run after ``09_augment_refusals.py``:

    python pipelines/10_safetune_mix.py
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN = PROJECT_ROOT / "data" / "splits" / "train.jsonl"
VAL = PROJECT_ROOT / "data" / "splits" / "val.jsonl"
TESTSET = PROJECT_ROOT / "eval" / "testset.jsonl"
SAFETY = PROJECT_ROOT / "eval" / "safety_set.jsonl"
MANIFEST = PROJECT_ROOT / "data" / "splits" / "safetune_manifest.json"

STAGE = "safetune-benign-v1"
ID_PREFIX = "gold_st_"
PER_TOOL = 12
NEWS_ROWS = 12
#: No training trajectory starts with a news call, so those rows are rebuilt
#: around a recorded news call instead of copied wholesale (see below).
PRIMARY_TOOLS = ("get_stock_quote", "get_financial_ratios", "get_historical_prices")
TOOLS = PRIMARY_TOOLS + ("get_company_news",)

#: Fresh phrasings: none equals a training, control or safety-set question.
TEMPLATES: dict[str, tuple[str, ...]] = {
    "get_stock_quote": (
        "How much does {ticker} cost per share at the moment?",
        "Quote {ticker} for me, please.",
        "What is the current share price of {ticker}?",
        "Where is {ticker} trading as of now?",
    ),
    "get_financial_ratios": (
        "Show me {ticker}'s valuation multiples.",
        "Pull {ticker}'s earnings and book multiples.",
        "What are {ticker}'s leverage and profitability ratios?",
        "Give me {ticker}'s P/E and return on equity figures.",
    ),
    "get_historical_prices": (
        "How has {ticker} traded over the past {period}?",
        "Show me {ticker}'s closes for the last {period}.",
        "What did {ticker} do over the past {period}?",
        "Chart {ticker} over the last {period}.",
    ),
    "get_company_news": (
        "What news is out about {ticker}?",
        "Catch me up on {ticker} headlines.",
        "Any press coverage on {ticker} recently?",
        "What are the latest stories on {ticker}?",
    ),
}

PERIOD_WORDS = {
    "1mo": "month",
    "3mo": "three months",
    "6mo": "six months",
    "1y": "year",
    "2y": "two years",
    "5y": "five years",
}

ABSTAIN_BEHAVIORS = frozenset(
    {
        "refuse_advice",
        "abstain_invalid_ticker",
        "abstain_malformed_response",
        "stop_after_bounded_attempts",
    }
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_tool(row: dict[str, Any]) -> str | None:
    try:
        return json.loads(row["messages"][2]["content"]).get("tool")
    except (IndexError, KeyError, json.JSONDecodeError):
        return None


def pick_sources(train: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """One source row per tool, twelve distinct tickers each."""
    picked: dict[str, list[dict[str, Any]]] = {tool: [] for tool in PRIMARY_TOOLS}
    used_tickers: dict[str, set[str]] = {tool: set() for tool in PRIMARY_TOOLS}
    for row in sorted(train, key=lambda item: item["id"]):
        if row.get("augmentation"):
            continue
        if row["expected_behavior"] in ABSTAIN_BEHAVIORS:
            continue
        tool = first_tool(row)
        if tool not in picked or len(picked[tool]) >= PER_TOOL:
            continue
        ticker = row["tickers"][0]
        if ticker in used_tickers[tool]:
            continue
        used_tickers[tool].add(ticker)
        picked[tool].append(row)
    short = {tool: len(rows) for tool, rows in picked.items() if len(rows) < PER_TOOL}
    if short:
        raise SystemExit(f"not enough source trajectories per tool: {short}")
    return picked


def pick_news_sources(train: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows that contain a recorded news call, one per distinct ticker."""
    picked: list[dict[str, Any]] = []
    used: set[str] = set()
    for row in sorted(train, key=lambda item: item["id"]):
        if row.get("augmentation") or row["expected_behavior"] in ABSTAIN_BEHAVIORS:
            continue
        ticker = row["tickers"][0]
        if ticker in used:
            continue
        if any('"tool": "get_company_news"' in message["content"] for message in row["messages"]):
            used.add(ticker)
            picked.append(row)
        if len(picked) == NEWS_ROWS:
            break
    if len(picked) < NEWS_ROWS:
        raise SystemExit(f"only {len(picked)} rows contain a news call")
    return picked


def news_trajectory(source: dict[str, Any], question: str) -> list[dict[str, Any]]:
    """System, question, one recorded news call, finish.

    The call id is renumbered to ``c1`` because the source row called news last;
    reordering without renumbering would hand the model an envelope whose id does
    not match the turn.
    """
    messages = source["messages"]
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        try:
            call = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        if call.get("tool") != "get_company_news" or index + 1 >= len(messages):
            continue
        result = json.loads(messages[index + 1]["content"])["result"]
        return [
            {"role": "system", "content": messages[0]["content"]},
            {"role": "user", "content": question},
            {"role": "assistant", "content": json.dumps(call, separators=(",", ":"))},
            {
                "role": "tool",
                "content": json.dumps({"call_id": "c1", "result": result}, sort_keys=True),
            },
            {"role": "assistant", "content": json.dumps({"tool": "finish", "args": {}})},
        ]
    raise AssertionError(f"no news call in {source['id']}")


def build_news_rows(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, source in enumerate(sources):
        ticker = source["tickers"][0]
        question = TEMPLATES["get_company_news"][position % 4].format(ticker=ticker)
        rows.append(
            {
                "id": f"{ID_PREFIX}{100 + position:04d}",
                "intent": "news",
                "mutation": "none",
                "language": "en",
                "tickers": [ticker],
                "period": source.get("period"),
                "question": question,
                "expected_behavior": "full_report",
                "messages": news_trajectory(source, question),
                "report": {
                    "question": question,
                    "behavior": "full_report",
                    "skeleton": {
                        "claims": [],
                        "missing": [],
                        "limitations": ["Values come from a recorded provider snapshot."],
                    },
                    "ledger": [],
                    "ledger_text": "",
                },
                "augmentation": STAGE,
            }
        )
    return rows


def build_rows(
    sources: dict[str, list[dict[str, Any]]],
    news_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 1
    for tool in PRIMARY_TOOLS:
        templates = TEMPLATES[tool]
        for position, source in enumerate(sources[tool]):
            row = copy.deepcopy(source)
            ticker = row["tickers"][0]
            period = PERIOD_WORDS.get(row.get("period") or "", "year")
            question = templates[position % len(templates)].format(
                ticker=ticker, period=period
            )
            row["id"] = f"{ID_PREFIX}{index:04d}"
            row["question"] = question
            row["messages"][1] = {"role": "user", "content": question}
            if isinstance(row.get("report"), dict):
                row["report"]["question"] = question
            row["augmentation"] = STAGE
            rows.append(row)
            index += 1
    rows.extend(build_news_rows(news_sources))
    return rows


def validate(
    base_train: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    val: list[dict[str, Any]],
    testset: list[dict[str, Any]],
    safety: list[dict[str, Any]],
) -> dict[str, Any]:
    assert len(rows) == PER_TOOL * len(TOOLS) == 48, len(rows)
    assert len({row["id"] for row in rows}) == len(rows), "duplicate ids"
    assert len({row["question"] for row in rows}) == len(rows), "duplicate questions"

    elsewhere = {
        "train": {row["question"] for row in base_train},
        "val": {row["question"] for row in val},
        "testset": {row["question"] for row in testset},
        "safety": {row["question"] for row in safety},
    }
    overlaps = {
        name: sorted(questions & {row["question"] for row in rows})
        for name, questions in elsewhere.items()
    }
    assert not any(overlaps.values()), f"benign rows overlap existing data: {overlaps}"

    val_tickers = {ticker for row in val for ticker in row["tickers"]}
    leaked = sorted({ticker for row in rows for ticker in row["tickers"]} & val_tickers)
    assert not leaked, f"validation tickers in benign rows: {leaked}"

    for row in rows:
        tool = first_tool(row)
        assert tool in TOOLS, f"{row['id']} does not start with a data tool: {tool}"
        assert row["expected_behavior"] not in ABSTAIN_BEHAVIORS, row["id"]
        assert row["messages"][-1]["content"].rstrip().endswith("}"), row["id"]

    by_tool: dict[str, int] = {}
    for row in rows:
        tool = first_tool(row) or "?"
        by_tool[tool] = by_tool.get(tool, 0) + 1
    return {
        "rows_added": len(rows),
        "by_tool": dict(sorted(by_tool.items())),
        "tickers": sorted({ticker for row in rows for ticker in row["tickers"]}),
        "overlap_with_existing": overlaps,
        "val_tickers_leaked": leaked,
    }


def main() -> int:
    train = load_jsonl(TRAIN)
    val = load_jsonl(VAL)
    testset = load_jsonl(TESTSET)
    safety = load_jsonl(SAFETY)

    base_train = [row for row in train if row.get("augmentation") != STAGE]
    rows = build_rows(pick_sources(base_train), pick_news_sources(base_train))
    report = validate(base_train, rows, val, testset, safety)

    digest_before = sha256(TRAIN)
    combined = base_train + rows
    write_jsonl(TRAIN, combined)
    refusals = sum(1 for row in combined if row["expected_behavior"] == "refuse_advice")
    MANIFEST.write_text(
        json.dumps(
            {
                "stage": STAGE,
                "source": "SafeTune: over-refusal repair after the insider pass",
                "train_rows_before": len(base_train),
                "train_rows_after": len(combined),
                "refusal_rows": refusals,
                "refusal_share": round(refusals / len(combined), 4),
                "sha256_before": digest_before,
                "sha256_after": sha256(TRAIN),
                **report,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(
        f"train: {len(base_train)} -> {len(combined)} rows; "
        f"refusals {refusals} ({refusals / len(combined):.1%} of the mix)"
    )
    print(f"Wrote {TRAIN.relative_to(PROJECT_ROOT)} and {MANIFEST.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
