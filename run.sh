#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT="${LILI_PORT:-8090}"
VENV=".venv"

if [ ! -d "$VENV" ]; then
  echo "[setup] Creating venv..."
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! python -c "import torch, diffusers, transformers, huggingface_hub" >/dev/null 2>&1; then
  echo "[setup] Installing dependencies (this may take a while)..."
  pip install --upgrade pip
  # CPU torch by default; set LILI_TORCH_INDEX to override (e.g. CUDA wheel index)
  TORCH_INDEX="${LILI_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
  pip install --extra-index-url "$TORCH_INDEX" -r requirements.txt
fi

echo "[run] Starting LiliPixel harness on http://127.0.0.1:${PORT}/"
exec python server.py
