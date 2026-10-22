#!/bin/bash
# Verify a real training step at the budget the data actually requires.
#
# Measured against artifacts/suites/real-v2 (kev's own numbers, from its failure message):
#   longest packed record 7714 tokens, longest row (state + branch) 4807.
# kev derives all three limits from --max_state alone (kev/model.py: training_context):
#   max_branch (the row) = N + 640,  max_packed = N + 1664.
# So the binding constraint is the packed one:
#   N >= 7744 - 1664 = 6080          (and  N >= 4807 - 640 = 4167, satisfied)
#   N = 6080 <= MAX_TRAIN_STATE 7552.
cd /mnt/f/E/hermes-workspace/scKevType || exit 1
# shellcheck disable=SC1091
source scripts/wsl/env.sh >/dev/null 2>&1

N=6080
OUT=/mnt/f/E/hermes-workspace/scKevType/reports/step10-train-verify.log
{
  echo "=== train verify at max_state=$N  $(date -Is) ==="
  echo
  echo "--- compute probe: VRAM and tokens/s at this budget ---"
  uv run python -m sckevtype.train.compute --models 0.8b --batches 1,2 \
    --suite artifacts/suites/real-v2 --max_state "$N" --warmup 1 --steps 3 \
    --dtype bf16 --device cuda 2>&1 | tail -20
  echo
  echo "--- a real training step on the real suite ---"
  rm -rf runs/step10-train-smoke
  uv run python -m sckevtype.train.train --suite artifacts/suites/real-v2 \
    --base Qwen/Qwen3.5-0.8B-Base \
    --base_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
    --steps 1 --batch 2 --accum 4 --dtype bf16 --device cuda \
    --max_state "$N" --out runs/step10-train-smoke 2>&1 | tail -30
  echo
  echo "--- what it wrote ---"
  find runs/step10-train-smoke -type f 2>/dev/null | while read -r f; do
    printf '  %10s  %s\n' "$(stat -c%s "$f")" "$f"
  done
  echo
  echo "=== DONE $(date -Is) ==="
} > "$OUT" 2>&1
