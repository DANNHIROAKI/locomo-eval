#!/usr/bin/env bash
set -euo pipefail

AGENT_MODEL="openai/gpt-4.1-mini"
STATE_DIR_NAME="locomo-bench"
STATE_DIR_ROOT="$PWD"
STATE_DIR="${STATE_DIR_ROOT%/}/${STATE_DIR_NAME}"
BENCH_ENV_FILE="${STATE_DIR}/openclaw.env"

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

openclaw config set agents.defaults.model.primary "${AGENT_MODEL}" >/dev/null

echo "Starting gateway with agent model hardcoded to: ${AGENT_MODEL}"
openclaw gateway run --bind loopback --force
