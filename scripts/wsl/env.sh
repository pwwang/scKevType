#!/usr/bin/env bash
# Source this before any sckevtype work in WSL2:  source scripts/wsl/env.sh
# Keeps the venv + data on the Linux filesystem (fast) while the source tree
# stays on /mnt/f so it is visible from Windows.

export SCKEVTYPE_ROOT=/mnt/f/E/hermes-workspace/scKevType
export UV_PROJECT_ENVIRONMENT="$HOME/venvs/sckevtype"
export SCKEVTYPE_DATA="$HOME/data/sckevtype"
export SCKEVTYPE_RUNS="$HOME/runs/sckevtype"
export SCKEVTYPE_ARTIFACTS="$SCKEVTYPE_ROOT/artifacts"

# claude (coding agent) + uv live in miniconda3 / ~/.local/bin
export PATH="$HOME/miniconda3/bin:$HOME/.local/bin:$PATH"

# --- HuggingFace -------------------------------------------------------------------
# The token lives in ~/.bashrc as HUGGINGFACE_KEY, but that line sits *after*
# .bashrc's non-interactive guard (`*) return;;`), so non-interactive shells —
# every script, every delegated agent, every training job — never inherit it.
# And HUGGINGFACE_KEY is not a name huggingface_hub reads: it checks HF_TOKEN /
# HUGGING_FACE_HUB_TOKEN. So read it from its home and publish it under the
# names the library actually honours.
if [ -z "${HF_TOKEN:-}" ]; then
  _hf_tok="$(sed -n 's/^[[:space:]]*export[[:space:]]*HUGGINGFACE_KEY=//p' \
             "$HOME/.bashrc" 2>/dev/null | tail -1 | tr -d "\"'")"
  if [ -n "$_hf_tok" ]; then
    export HF_TOKEN="$_hf_tok"
    export HUGGING_FACE_HUB_TOKEN="$_hf_tok"
  fi
  unset _hf_tok
fi
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
# -----------------------------------------------------------------------------------

mkdir -p "$SCKEVTYPE_DATA" "$SCKEVTYPE_RUNS"

cd "$SCKEVTYPE_ROOT" || return 1
