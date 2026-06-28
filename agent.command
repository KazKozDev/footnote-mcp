#!/bin/bash
set -e

cd "$(dirname "$0")"

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP_FILE="$VENV_DIR/.requirements.sha256"
PLAYWRIGHT_STAMP_FILE="$VENV_DIR/.playwright.chromium"

finish() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo ""
    echo "Command failed with exit code $status"
    read -r -p "Press Enter to close..."
  fi
}
trap finish EXIT

if [ ! -f "requirements.txt" ]; then
  echo "requirements.txt not found in $(pwd)"
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

CURRENT_REQUIREMENTS_HASH="$(python - <<'PY'
from pathlib import Path
import hashlib

path = Path("requirements.txt")
print(hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "")
PY
)"

INSTALLED_REQUIREMENTS_HASH=""
if [ -f "$STAMP_FILE" ]; then
  INSTALLED_REQUIREMENTS_HASH="$(cat "$STAMP_FILE")"
fi

if [ "$CURRENT_REQUIREMENTS_HASH" != "$INSTALLED_REQUIREMENTS_HASH" ]; then
  echo "Installing Python dependencies..."
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  echo "$CURRENT_REQUIREMENTS_HASH" > "$STAMP_FILE"
fi

if [ ! -f "$PLAYWRIGHT_STAMP_FILE" ]; then
  echo "Installing Playwright Chromium browser..."
  python -m playwright install chromium
  touch "$PLAYWRIGHT_STAMP_FILE"
fi

if [ "${WEBOPERATOR_SETUP_ONLY:-0}" = "1" ]; then
  echo "Virtual environment is ready."
  exit 0
fi

# ── Ollama server ────────────────────────────────────────────────────────────
OLLAMA_BIN="/opt/homebrew/bin/ollama"
if [ ! -x "$OLLAMA_BIN" ]; then
  OLLAMA_BIN="$(which ollama 2>/dev/null || true)"
fi

if [ -n "$OLLAMA_BIN" ]; then
  # Stop any existing instance
  osascript -e 'quit app "Ollama"' 2>/dev/null || true
  pkill -f "ollama serve" 2>/dev/null || true
  sleep 1

  export OLLAMA_USE_MLX=1
  export OLLAMA_FLASH_ATTENTION=1
  export OLLAMA_KV_CACHE_TYPE=q8_0
  export OLLAMA_KEEP_ALIVE=-1
  export OLLAMA_NUM_PARALLEL=2
  export OLLAMA_MAX_LOADED_MODELS=2
  export OLLAMA_CONTEXT_LENGTH=64000
  export OLLAMA_HOST=0.0.0.0:11434

  "$OLLAMA_BIN" serve &>/tmp/ollama_serve.log &
  OLLAMA_PID=$!

  # Wait for server to be ready
  echo "Starting Ollama server..."
  for i in $(seq 1 20); do
    if curl -sf http://localhost:11434/api/tags &>/dev/null; then
      echo "Ollama ready."
      break
    fi
    sleep 1
  done

fi

python agent/agent.py

echo ""
read -r -p "Press Enter to close..."
