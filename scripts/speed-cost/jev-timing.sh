#!/usr/bin/env bash
# The Jev arm of the speed/cost pass: the live TypeSafe System One endpoint, on the same 200-cell draw
# the local engines are timed on, from this box (no GPU, no cluster: Jev is served remotely).
#
#   wsl.exe -d Ubuntu-24.04 -e bash -ic 'cd /mnt/f/E/hermes-workspace/scKevType && \
#       source scripts/wsl/env.sh && ./scripts/speed-cost/jev-timing.sh'
#
# `bash -lc`/`bash -c 'source ~/.bashrc'` do NOT see the key — the repo keeps it in ~/.bashrc, which
# early-returns for a non-interactive shell — so this must be an *interactive* shell. Because the key
# is read from the environment and never written anywhere, and the read is the only thing about this
# script that touches it, the key's length is printed and its value never is.
#
# Spend, bounded by the client's own caps: one ask per record of the 200-cell draw (800 calls) plus
# the abstain variants the convention adds, under a $1.00 budget, a $0.10 single-call cap and the
# vendor's own usage fields as the only token authority. `ask` is resumable per (cell, family, ask),
# so a re-run of this script against the same OUT spends nothing and rewrites the same rows.
#
# OUT/workers are the experiment's parameters: WORKERS=1 is the sequential pass (clean per-ask
# latency), WORKERS=4 is the in-flight pass the earlier smoke run used (throughput under concurrency).
set -euo pipefail

if [ -z "${JEV_KEY:-}" ]; then
  echo "JEV_KEY is not set in this shell, so nothing was sent." >&2
  echo "  it lives in ~/.bashrc, which only *interactive* shells read; run:" >&2
  echo "  wsl.exe -d Ubuntu-24.04 -e bash -ic 'cd /mnt/f/E/hermes-workspace/scKevType && ./scripts/speed-cost/jev-timing.sh'" >&2
  exit 2
fi
echo "JEV_KEY: present, ${#JEV_KEY} chars (value never printed, never written)"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/wsl/env.sh

OUT=${OUT:-reports/speed-cost/jev}
WORKERS=${WORKERS:-1}
CELLS_PER_DATASET=${CELLS_PER_DATASET:-50}
SEED=${SEED:-20260929}
BUDGET_USD=${BUDGET_USD:-1.0}
SLUG=${SLUG:-jev}
SUMMARY=${SUMMARY:-reports/speed-cost/$SLUG.json}
REFERENCE=${REFERENCE:-reports/jev-smoke/sample.json}
mkdir -p "$OUT"

echo
echo "=== 1/4 sample: the same 200 cells the smoke draw used (no network, no key use) ==="
uv run --no-sync python -m sckevtype.jev.probe sample \
  --out "$OUT" --split test --cells-per-dataset "$CELLS_PER_DATASET" --seed "$SEED"
uv run --no-sync python - "$REFERENCE" "$OUT/sample.json" <<'PY'
import json, sys
reference, produced = (json.load(open(path)) for path in sys.argv[1:3])
same = set(reference["records"]) == set(produced["records"])
print(f"  population: {produced['cells']} cells, {len(produced['records'])} records, "
      f"{produced.get('questions')} questions; identical record set to {sys.argv[1]}: {same}")
if not same:
    raise SystemExit("the draw is not the smoke draw; the table would not be on one population")
PY

echo
echo "=== 2/4 ask: serve the sample (network, budgeted, resumable) — workers=$WORKERS ==="
START=$(date +%s)
uv run --no-sync python -m sckevtype.jev.probe ask \
  --out "$OUT" --workers "$WORKERS" --budget-usd "$BUDGET_USD" \
  --max-call-usd 0.10 --max-calls 2000 --progress-every 50
ASK_WALL=$(( $(date +%s) - START ))
echo "  ask wall: ${ASK_WALL}s"

echo
echo "=== 3/4 evaluate: score the saved answers (local, free) ==="
uv run --no-sync python -m sckevtype.jev.probe evaluate --out "$OUT"

echo
echo "=== 4/4 collect: latency, tokens and the vendor's own price, from the raw dumps ==="
uv run --no-sync python scripts/speed-cost/collect_timing.py jev --raw "$OUT/raw" --rows "$OUT/rows.jsonl" \
  --ask-log "$OUT/ask-log.json" --engine "$SLUG" --engine-id "TypeSafe System One / ${JEV_MODEL:-jev-latest}" \
  --gpus 0 --device-name "none (hosted API)" --job-id "none (local WSL)" \
  --groups-from "$REFERENCE" --wall-s "$ASK_WALL" --out "$SUMMARY"
uv run --no-sync python - "$SUMMARY" "$OUT/ask-log.json" "${WORKERS:-1}" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
workers = sys.argv[3] if len(sys.argv) > 3 else "1"
accounting = (json.load(open(sys.argv[2])) or {}).get("accounting", {})
measured = summary.get("measured") or {}
print("  served:", accounting.get("served_models"), "calls:", accounting.get("calls"),
      "in-tokens:", accounting.get("input_tokens"), "out-tokens:", accounting.get("output_tokens"))
print(f"  spend at the vendor's listed ${accounting.get('listed_input_usd_per_million')}/M input tokens: "
      f"${accounting.get('estimated_usd'):.4f}  (price source: {accounting.get('price_source')})")
if measured:
    print(f"  median {measured['latency_ms_per_ask']['median']:.1f} ms/ask, "
          f"p95 {measured['latency_ms_per_ask']['p95']:.1f} ms/ask, "
          f"{measured['tokens_in_per_ask']['mean']:.0f} in-tokens/ask, "
          f"{measured['tokens_out_per_ask']['mean']:.1f} out-tokens/ask, "
          f"{measured['asks_per_s']:.2f} asks/s at workers={workers}")
PY
echo "wrote $SUMMARY"
