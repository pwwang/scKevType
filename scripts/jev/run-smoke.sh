#!/usr/bin/env bash
#
# The whole costed Jev smoke test, in one command: draw 200 cells, serve them, score them, and
# compare against the shipped eval report.
#
# JEV_KEY comes from the environment and is never written to disk. The repo keeps it in ~/.bashrc,
# which only *interactive* shells read, so `bash -lc` will not see it even if your login shell can:
#
#   wsl.exe -e bash -ic 'cd /mnt/f/E/hermes-workspace/scKevType && ./scripts/jev/run-smoke.sh'
#
# Spend, measured on the 2026-09-29 run: 796 calls, 2.90 M input tokens, $0.1219, 2m00s.
# `ask` is resumable — it re-reads reports/jev-smoke/raw/ and only pays for what is missing.
set -euo pipefail

if [ -z "${JEV_KEY:-}" ]; then
  echo "JEV_KEY is not set in this shell." >&2
  echo "It lives in ~/.bashrc, which only interactive shells read; run this as:" >&2
  echo "  wsl.exe -e bash -ic 'cd /mnt/f/E/hermes-workspace/scKevType && ./scripts/jev/run-smoke.sh'" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/wsl/env.sh

OUT="${OUT:-reports/jev-smoke}"
CELLS_PER_DATASET="${CELLS_PER_DATASET:-50}"   # 4 datasets x 50 = the 200-cell probe
SEED="${SEED:-20260929}"
WORKERS="${WORKERS:-4}"                        # threads in one process, not processes
BUDGET_USD="${BUDGET_USD:-1.0}"

echo "=== 1/4 sample: draw the cells (no network, no key) ==="
uv run --no-sync python -m sckevtype.jev.probe sample \
  --out "$OUT" --split test --cells-per-dataset "$CELLS_PER_DATASET" --seed "$SEED"

echo "=== 2/4 ask: serve the sample (network; resumable; budgeted) ==="
uv run --no-sync python -m sckevtype.jev.probe ask \
  --out "$OUT" --workers "$WORKERS" --budget-usd "$BUDGET_USD" \
  --max-call-usd 0.10 --max-calls 2000

echo "=== 3/4 evaluate: score the saved answers (local) ==="
uv run --no-sync python -m sckevtype.jev.probe evaluate --out "$OUT"

echo "=== 4/4 compare: side by side with the shipped eval report ==="
uv run --no-sync python scripts/jev/compare.py

echo "=== the axis-1 dump, read by the house tool (must report 0 reconstruction problems) ==="
# inside the repo on purpose: `uv run` finds the project by walking up, and a bare mktemp dir would
# import no sckevtype at all
TMP="$OUT/axis1-demo"
rm -rf "$TMP"
mkdir -p "$TMP/reports"
cp "$OUT/axis1-rows.jsonl" "$TMP/reports/axis1-rows-eval.jsonl"
( cd "$TMP" && uv run --no-sync python "$OLDPWD/scripts/axis1-recompute.py" )
rm -rf "$TMP"
