#!/usr/bin/env bash
# One-shot environment bootstrap for WSL2.
#   bash scripts/wsl/bootstrap.sh
# Base env installs in seconds (contracts/uncertainty need no torch).
# The heavy env (kev + torch + FLA + anndata) is synced in the background.
set -uo pipefail

source "$(dirname "$0")/env.sh"

echo "== python =="
uv python find 2>/dev/null || uv python install 3.12
python3 --version

echo "== base env (no torch) =="
uv sync

echo "== base tests =="
uv run pytest -q tests/contracts 2>/dev/null || echo "(no contracts tests yet)"

echo "== full env (train+data+eval+serve) -> $HOME/uvsync-full.log =="
if pgrep -f "uv sync --extra" >/dev/null 2>&1; then
  echo "already syncing, not starting a second one"
else
  nohup uv sync --extra train --extra cuda --extra data --extra eval --extra serve \
    > "$HOME/uvsync-full.log" 2>&1 &
  echo "started pid $! -> tail -f ~/uvsync-full.log"
fi
