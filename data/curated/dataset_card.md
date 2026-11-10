# Dataset Card: finance_agent_ws4

## Summary

- **Run timestamp**: 2026-11-10T12:22:28.074417Z
- **Wall clock time**: 47.7s
- **Pipeline config hash**: `afeaaf95fd22f5cc`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 800 | 800 | — | 0 | 800 |
| SchemaGate | 800 | 262 | 0 | 538 | 262 |
| ExactDeduplicator | 262 | 262 | — | — | 262 |
| MinHashDeduplicator | 262 | 170 | — | — | 170 |
| TextCleaner | 170 | 170 | — | — | 170 |
| MessagesExporter | — | 170 | — | — | 170 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 800/0 passed (N/A), 0 rejected
- **SchemaGate**: 262/800 passed (32.8%), 538 rejected


**Rejection reasons:**
  - `below_min_tokens:9`: 142
  - `below_min_tokens:8`: 139
  - `below_min_tokens:7`: 139
  - `below_min_tokens:6`: 117
  - `below_min_tokens:5`: 1

## Deduplication

  - near_duplicates_removed: 92
  - surviving_count: 170

## Known Limitations

- MinHash threshold used: **0.85** — this is a starting point, not a universal answer. Adjust per domain (legal, medical, and other domain-specific corpora may need a lower threshold to avoid losing structurally similar but semantically valid examples).
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.23
- **python**: 3.11.9
