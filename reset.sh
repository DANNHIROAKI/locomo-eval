#!/usr/bin/env bash
set -euo pipefail

FACTORY_RESET=false
STATE_DIR_NAME="locomo-bench"
STATE_DIR_ROOT="$PWD"
STATE_DIR="${STATE_DIR_ROOT%/}/${STATE_DIR_NAME}"
BENCH_CONFIG_PATH="${STATE_DIR}/openclaw.json"
ENV_FILE="${STATE_DIR}/openclaw.env"
WORKSPACE_SHADOW_PLUGIN_DIR="${STATE_DIR}/workspace/.openclaw/extensions/memory-lancedb"
GLOBAL_STATE_DIR="${HOME}/.openclaw"
GLOBAL_CONFIG_PATH="${GLOBAL_STATE_DIR}/openclaw.json"

echo "Reset: stopping gateway if it is running..."
openclaw gateway stop >/dev/null 2>&1 || true

echo "Reset: clearing any process still listening on port 18789..."
PORT_PIDS="$(lsof -ti tcp:18789 || true)"
if [[ -n "${PORT_PIDS}" ]]; then
  kill ${PORT_PIDS} >/dev/null 2>&1 || true
fi

echo "Reset: removing benchmark-local state..."
rm -rf "${STATE_DIR}"
rm -rf "${WORKSPACE_SHADOW_PLUGIN_DIR}"
rm -f "${BENCH_CONFIG_PATH}"
rm -f "${ENV_FILE}"

if [[ "${FACTORY_RESET}" == "true" ]]; then
  echo "Reset: performing factory reset of global OpenClaw config..."
  if [[ -f "${GLOBAL_CONFIG_PATH}" ]]; then
    BACKUP_PATH="${GLOBAL_CONFIG_PATH}.bak.$(date +%Y%m%d-%H%M%S)"
    mv "${GLOBAL_CONFIG_PATH}" "${BACKUP_PATH}"
  fi
  openclaw onboard
else
  echo "Reset: restoring global config to memory-core baseline..."
  openclaw config unset plugins.entries.memory-lancedb >/dev/null 2>&1 || true
  openclaw config unset plugins.allow >/dev/null 2>&1 || true
  openclaw config set plugins.slots.memory memory-core >/dev/null
fi

cat <<EOF
OpenClaw benchmark reset complete.

Removed benchmark state:
  ${STATE_DIR}

Removed old workspace shadow plugin copy:
  ${WORKSPACE_SHADOW_PLUGIN_DIR}

Removed benchmark-local config, if present:
  ${BENCH_CONFIG_PATH}

Removed benchmark env file, if present:
  ${ENV_FILE}

Current reset mode:
  FACTORY_RESET=${FACTORY_RESET}

If FACTORY_RESET=false, the global config was kept and reset to memory-core.
If FACTORY_RESET=true, the global config was moved aside and OpenClaw was re-onboarded.
EOF
