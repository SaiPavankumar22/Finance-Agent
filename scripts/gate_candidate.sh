#!/usr/bin/env bash
# Gate an AlignTune run: wait for the adapter, score the three eval sets, write
# the report. Every stage skips when its artifact already exists, so a dropped
# shell or a restarted runner resumes from the first missing stage instead of
# re-spending GPU time.
#
#   nohup bash scripts/gate_candidate.sh aligntune-v5 aligntune-r32-e4-v5 \
#     > artifacts/sft/aligntune-v5-gates.log 2>&1 &
set -u

TAG=${1:?tag, e.g. aligntune-v5}
OUT=${2:?volume directory, e.g. aligntune-r32-e4-v5}
ADAPTER="/out/${OUT}"

export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MSYS_NO_PATHCONV=1
cd "$(dirname "$0")/.." || exit 1

echo "== waiting for ${ADAPTER}/adapter_model.safetensors"
for _ in $(seq 1 240); do
  if modal volume ls finance-agent-sft-adapters "$OUT" 2>/dev/null | grep -q adapter_model.safetensors; then
    echo "adapter present"
    break
  fi
  sleep 30
done
if ! modal volume ls finance-agent-sft-adapters "$OUT" 2>/dev/null | grep -q adapter_model.safetensors; then
  echo "FAILED: adapter never appeared in ${ADAPTER}"
  exit 1
fi

stage() { # label, artifact, adapter-var, tag-var, entrypoint
  local label=$1 artifact=$2 avar=$3 tvar=$4 entry=$5
  if [ -f "$artifact" ]; then
    echo "skip ${label}: have ${artifact}"
    return 0
  fi
  echo "== ${label}"
  if ! env "${avar}=${ADAPTER}" "${tvar}=${TAG}" modal run "training/train_full.py::${entry}"; then
    echo "FAILED ${label}"
    exit 1
  fi
}

stage "frozen eval" "artifacts/sft/eval-${TAG}.json" SFT_ADAPTER SFT_TAG eval_main
stage "safety eval" "artifacts/sft/safety-${TAG}.json" SAFETY_ADAPTER SAFETY_TAG safety_main
stage "india eval" "artifacts/sft/india-${TAG}.json" INDIA_ADAPTER INDIA_TAG india_main

echo "== gates"
python -m pipelines.13_gates --candidate-tag "$TAG" --report "artifacts/safety/gates-${TAG}.md" || exit 1
echo "ALL_DONE"
