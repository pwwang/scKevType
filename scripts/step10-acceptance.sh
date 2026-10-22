#!/bin/bash
# Step 10 acceptance: the row measurement, the compute probe at the DERIVED budget, and a training step.
# kev couples the budgets (kev/model.py training_context): branch = 1024 + (N-384) = N+640,
# packed = 2048 + (N-384) = N+1664. `max_branch` IS the training row — "the row (state + one branch)"
# — and that is the binding check (verified: at N=3200 the row limit 3840 rejected a 4488-token row,
# while the packed limit 4864 would have allowed it).
#
#   train max row = 4807  ->  N >= 4807 - 640 = 4167  ->  4224;  one step of margin: N = 4288
#   check: 4288 + 640 = 4928 >= 4807
cd /mnt/f/E/hermes-workspace/scKevType || exit 1
# shellcheck disable=SC1091
source scripts/wsl/env.sh >/dev/null 2>&1

OUT=/mnt/f/E/hermes-workspace/scKevType/reports/step10-acceptance.log
{
  echo "=== step10 acceptance $(date -Is) ==="
  echo
  echo "--- whole train split: true row maxima ---"
  uv run python scripts/measure-row-tokens.py artifacts/suites/real-v2 6000 train 2>&1 \
    | grep -E "^family|^[a-z_]+ +[0-9]|^max row|^median row|N >=|ceiling"
  echo
  echo "--- whole calibration split: true row maxima ---"
  uv run python scripts/measure-row-tokens.py artifacts/suites/real-v2 3000 calibration 2>&1 \
    | grep -E "^max row|^median row|N >=|ceiling"
  echo
  echo "--- acceptance 5: compute probe at max_state 4288 (0.8b, bf16, cuda) ---"
  uv run python -m sckevtype.train.compute --models 0.8b --batches 1,2 \
    --suite artifacts/suites/real-v2 --max_state 4288 --warmup 1 --steps 3 \
    --dtype bf16 --device cuda 2>&1 | tail -22
  echo
  echo "--- acceptance 6: a real training step on the real suite ---"
  rm -rf runs/step10-train-smoke
  uv run python -m sckevtype.train.train --suite artifacts/suites/real-v2 \
    --base Qwen/Qwen3.5-0.8B-Base \
    --base_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
    --steps 1 --batch 2 --accum 4 --dtype bf16 --device cuda \
    --max_state 4288 --out runs/step10-train-smoke 2>&1 | tail -25
  echo "  trainer rc=$?"
  echo "--- what it wrote ---"
  find runs/step10-train-smoke -type f 2>/dev/null | while read -r f; do
    printf '  %10s  %s\n' "$(stat -c%s "$f")" "$f"
  done
  echo
  echo "=== DONE $(date -Is) ==="
} > "$OUT" 2>&1
