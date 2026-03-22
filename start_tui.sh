#!/usr/bin/env bash
set -euo pipefail

STATE_DIR_NAME="locomo-bench"
STATE_DIR_ROOT="$PWD"
STATE_DIR="${STATE_DIR_ROOT%/}/${STATE_DIR_NAME}"
BENCH_ENV_FILE="${STATE_DIR}/openclaw.env"
SESSION_KEY="${1:-locomo-debug}"

if [[ ! -f "${BENCH_ENV_FILE}" ]]; then
  echo "Missing ${BENCH_ENV_FILE}. Run setup_memory_core.sh, setup_memory_lancedb.sh, or setup_memory_lancedb_pro.sh first." >&2
  exit 1
fi

set -a
source "${BENCH_ENV_FILE}"
if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  source .env
fi
set +a

if [[ -z "${OPENCLAW_GATEWAY_TOKEN:-}" ]]; then
  echo "OPENCLAW_GATEWAY_TOKEN is missing. Re-run the setup script for your chosen backend." >&2
  exit 1
fi

echo "Starting OpenClaw TUI"
echo "  session: ${SESSION_KEY}"
echo "  config: ${OPENCLAW_CONFIG_PATH}"
echo "  state: ${OPENCLAW_STATE_DIR}"
echo "  gateway: ws://127.0.0.1:18789"

openclaw tui \
  --url ws://127.0.0.1:18789 \
  --token "${OPENCLAW_GATEWAY_TOKEN}" \
  --session "${SESSION_KEY}"
