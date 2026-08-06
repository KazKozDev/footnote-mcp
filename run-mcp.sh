#!/bin/bash
# Launch footnote-mcp for Cursor with project .env loaded.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
exec .venv/bin/footnote-mcp "$@"
