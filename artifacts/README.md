# Artifact policy

`artifacts/` contains reproducible evidence, not runtime source code.

- `sft/` contains training, evaluation, and reproduction payloads.
- `safety/` contains generated gate reports.
- `release/` contains the small, human-readable manifest for the accepted
  release.
- Historical experiment artifacts remain available for audit and are never
  used by deployment defaults.

Large adapter weights, caches, Modal volumes, and transient logs are published
to Hugging Face or Modal and are intentionally excluded from Git. Regenerate
local evaluation outputs with the commands in `README.md` and publish durable
copies to the evaluation repository.

