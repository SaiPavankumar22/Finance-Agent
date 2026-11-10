# Dataset Card: finance_agent_ws4

## Summary

- **Run timestamp**: 2026-11-10T14:51:51.135065Z
- **Wall clock time**: 0.2s
- **Pipeline config hash**: `55b87c207fbab640`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 800 | 800 | — | 0 | 800 |
| SchemaGate | 800 | 800 | 0 | 0 | 800 |
| ExactDeduplicator | 800 | 800 | — | — | 800 |
| TextCleaner | 800 | 800 | — | — | 800 |
| MessagesExporter | — | 800 | — | — | 800 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 800/0 passed (N/A), 0 rejected
- **SchemaGate**: 800/800 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - exact_duplicates_removed: 0

## Known Limitations

- No MinHash deduplication applied.
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
