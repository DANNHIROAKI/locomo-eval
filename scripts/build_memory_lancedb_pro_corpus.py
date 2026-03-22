from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory_lancedb import count_lancedb_rows, count_lancedb_rows_at_path, resolve_lancedb_config
from src.openclaw_cli import resolve_memory_slot, run_openclaw_command


def _benchmark_env_path() -> str | None:
    candidate = Path("locomo-bench/openclaw.env")
    if candidate.exists():
        return str(candidate)
    return None


load_dotenv()
load_dotenv(_benchmark_env_path(), override=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the full prebuilt memory-lancedb-pro corpus from an existing memory-lancedb store"
    )
    parser.add_argument(
        "--source-db",
        default="locomo-bench/lancedb",
        help="Path to the prebuilt memory-lancedb database directory",
    )
    parser.add_argument(
        "--manifest",
        default="locomo-bench/prebuilt-memory-lancedb-pro.json",
        help="Path to write the corpus build manifest",
    )
    args = parser.parse_args()

    slot = resolve_memory_slot()
    if slot != "memory-lancedb-pro":
        raise ValueError(
            f"OpenClaw memory slot is '{slot}', but corpus build requires 'memory-lancedb-pro'."
        )

    source_db = Path(args.source_db).resolve()
    if not source_db.exists():
        raise ValueError(f"Source memory-lancedb store does not exist: {source_db}")
    source_row_count = count_lancedb_rows_at_path(source_db)
    if not source_row_count:
        raise ValueError(f"Source memory-lancedb store is empty: {source_db}")

    run_openclaw_command(["openclaw", "memory-pro", "delete-bulk", "--scope", "global"])
    migrate_stdout = run_openclaw_command(
        [
            "openclaw",
            "memory-pro",
            "migrate",
            "run",
            "--source",
            str(source_db),
            "--default-scope",
            "global",
        ]
    )

    config = resolve_lancedb_config("memory-lancedb-pro")
    row_count = count_lancedb_rows(config)
    manifest = {
        "backend": "memory-lancedb-pro",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_db": str(source_db),
        "source_row_count": source_row_count,
        "db_path": str(config.db_path),
        "row_count": row_count,
        "migration_stdout": migrate_stdout.strip(),
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
