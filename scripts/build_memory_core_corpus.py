from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import build_memory_documents, load_locomo_samples
from src.memory_core import (
    prepare_memory_root,
    reindex_memory,
    resolve_memory_index_paths,
    resolve_memory_status,
    write_memory_documents,
)
from src.openclaw_cli import resolve_memory_slot


def _benchmark_env_path() -> str | None:
    candidate = Path("locomo-bench/openclaw.env")
    if candidate.exists():
        return str(candidate)
    return None


load_dotenv()
load_dotenv(_benchmark_env_path(), override=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full prebuilt memory-core corpus")
    parser.add_argument("--input", required=True, help="Path to locomo10.json")
    parser.add_argument(
        "--manifest",
        default="locomo-bench/prebuilt-memory-core.json",
        help="Path to write the corpus build manifest",
    )
    parser.add_argument(
        "--agent-id",
        default="main",
        help="OpenClaw agent id used for workspace and memory indexing",
    )
    args = parser.parse_args()

    slot = resolve_memory_slot()
    if slot != "memory-core":
        raise ValueError(
            f"OpenClaw memory slot is '{slot}', but corpus build requires 'memory-core'."
        )

    samples = load_locomo_samples(args.input)
    documents = []
    for sample in samples:
        documents.extend(build_memory_documents(sample))

    workspace_dir, db_path = resolve_memory_index_paths(args.agent_id)
    prepare_memory_root(workspace_dir, "memory/locomo")
    write_memory_documents(workspace_dir, documents)
    reindex_stdout = reindex_memory(args.agent_id)
    status = resolve_memory_status(args.agent_id)

    manifest = {
        "backend": "memory-core",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_path": args.input,
        "sample_count": len(samples),
        "document_count": len(documents),
        "workspace_dir": str(workspace_dir),
        "db_path": str(db_path),
        "files": status.files,
        "chunks": status.chunks,
        "reindex_stdout": reindex_stdout.strip(),
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
