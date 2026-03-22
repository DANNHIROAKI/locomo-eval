#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

STATE_DIR_NAME="locomo-bench"
STATE_DIR_ROOT="$PWD"
STATE_DIR="${STATE_DIR_ROOT%/}/${STATE_DIR_NAME}"
AGENT_ID="main"
BENCH_CONFIG_PATH="${STATE_DIR}/openclaw.json"
ENV_FILE="${STATE_DIR}/openclaw.env"

WORKSPACE_DIR="${STATE_DIR}/workspace"
SQLITE_DIR="${STATE_DIR}/sqlite"
LANCEDB_DIR="${STATE_DIR}/lancedb"
LANCEDB_PRO_DIR="${STATE_DIR}/lancedb-pro"
PLUGIN_EXTENSIONS_DIR="${WORKSPACE_DIR}/.openclaw/extensions"
LOCAL_PLUGIN_DIR="${PLUGIN_EXTENSIONS_DIR}/memory-lancedb"
LOCAL_PLUGIN_PRO_DIR="${PLUGIN_EXTENSIONS_DIR}/memory-lancedb-pro"

mkdir -p "${STATE_DIR}" "${WORKSPACE_DIR}" "${SQLITE_DIR}" "${LANCEDB_DIR}" "${LANCEDB_PRO_DIR}" "${PLUGIN_EXTENSIONS_DIR}"

export OPENCLAW_STATE_DIR="${STATE_DIR}"
export OPENCLAW_CONFIG_PATH="${BENCH_CONFIG_PATH}"

rm -rf "${LOCAL_PLUGIN_DIR}"
rm -rf "${LOCAL_PLUGIN_PRO_DIR}"

EXISTING_TOKEN=""
if [[ -f "${BENCH_CONFIG_PATH}" ]]; then
  EXISTING_TOKEN="$(sed -n 's/.*"token": "\(.*\)".*/\1/p' "${BENCH_CONFIG_PATH}" | head -n 1)"
fi

if [[ -n "${EXISTING_TOKEN}" ]]; then
  GATEWAY_TOKEN="${EXISTING_TOKEN}"
elif command -v openssl >/dev/null 2>&1; then
  GATEWAY_TOKEN="$(openssl rand -hex 24)"
else
  GATEWAY_TOKEN="$(uuidgen | tr '[:upper:]' '[:lower:]' | tr -d '-')"
fi

cat > "${BENCH_CONFIG_PATH}" <<EOF
{
  "env": {
    "OPENAI_API_KEY": "\${OPENAI_API_KEY}"
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "openai/gpt-4.1-mini"
      },
      "workspace": "${WORKSPACE_DIR}",
      "memorySearch": {
        "store": {
          "path": "${SQLITE_DIR}/{agentId}.sqlite"
        }
      }
    }
  },
  "gateway": {
    "mode": "local",
    "bind": "loopback",
    "auth": {
      "mode": "token",
      "token": "${GATEWAY_TOKEN}"
    },
    "http": {
      "endpoints": {
        "responses": {
          "enabled": true
        }
      }
    }
  },
  "plugins": {
    "allow": [
      "memory-core"
    ],
    "slots": {
      "memory": "memory-core"
    }
  }
}
EOF

cat > "${ENV_FILE}" <<EOF
OPENCLAW_STATE_DIR="${OPENCLAW_STATE_DIR}"
OPENCLAW_CONFIG_PATH="${OPENCLAW_CONFIG_PATH}"
OPENCLAW_GATEWAY_TOKEN="${GATEWAY_TOKEN}"
OPENAI_API_KEY="${OPENAI_API_KEY}"
EOF

cat <<EOF
OpenClaw benchmark state configured for memory-core.

OPENCLAW_STATE_DIR=${OPENCLAW_STATE_DIR}
OPENCLAW_CONFIG_PATH=${OPENCLAW_CONFIG_PATH}
OPENCLAW_GATEWAY_TOKEN=${GATEWAY_TOKEN}
AGENT_ID=${AGENT_ID}

Workspace memory:
  ${WORKSPACE_DIR}

memory-core SQLite index:
  ${SQLITE_DIR}/${AGENT_ID}.sqlite

memory-lancedb store:
  ${LANCEDB_DIR}

memory-lancedb-pro store:
  ${LANCEDB_PRO_DIR}

Session transcripts:
  ${STATE_DIR}/agents/${AGENT_ID}/sessions

Default agent model:
  openai/gpt-4.1-mini

To switch to the LanceDB backend:
  ./setup_memory_lancedb.sh

Next step:
  ./start_gateway.sh
EOF
