from __future__ import annotations

import json
import sqlite3
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.openclaw_cli import OpenClawCliError, load_openclaw_json, run_openclaw_command
from src.schema import MemoryChunk, MemoryDocument


class MemoryCoreError(RuntimeError):
    """Raised when memory-core setup or indexing fails."""


@dataclass(frozen=True)
class MemoryStatus:
    agent_id: str
    workspace_dir: Path
    db_path: Path | None
    files: int
    chunks: int
    backend: str
    sources: list[str]
    raw: dict[str, Any]


def resolve_memory_status(agent_id: str) -> MemoryStatus:
    try:
        payload = load_openclaw_json(["openclaw", "memory", "status", "--agent", agent_id, "--json"])
    except OpenClawCliError as exc:
        raise MemoryCoreError(str(exc)) from exc

    if not isinstance(payload, list) or not payload:
        raise MemoryCoreError("OpenClaw memory status did not return any agent entries")

    entry = payload[0]
    status = entry.get("status", {})
    workspace_dir = status.get("workspaceDir")
    if not isinstance(workspace_dir, str) or not workspace_dir.strip():
        raise MemoryCoreError("OpenClaw memory status did not include workspaceDir")

    db_path = status.get("dbPath")
    backend = str(status.get("backend", ""))
    return MemoryStatus(
        agent_id=str(entry.get("agentId", agent_id)),
        workspace_dir=Path(workspace_dir).expanduser(),
        db_path=Path(db_path).expanduser() if isinstance(db_path, str) and db_path.strip() else None,
        files=int(status.get("files", 0) or 0),
        chunks=int(status.get("chunks", 0) or 0),
        backend=backend,
        sources=[str(source) for source in status.get("sources", [])],
        raw=entry,
    )


def resolve_memory_index_paths(agent_id: str) -> tuple[Path, Path]:
    try:
        workspace = load_openclaw_json(["openclaw", "config", "get", "agents.defaults.workspace", "--json"])
        store_path = load_openclaw_json(
            ["openclaw", "config", "get", "agents.defaults.memorySearch.store.path", "--json"]
        )
    except OpenClawCliError as exc:
        raise MemoryCoreError(str(exc)) from exc

    if not isinstance(workspace, str) or not workspace.strip():
        raise MemoryCoreError("OpenClaw config did not include agents.defaults.workspace")
    if not isinstance(store_path, str) or not store_path.strip():
        raise MemoryCoreError("OpenClaw config did not include agents.defaults.memorySearch.store.path")

    workspace_dir = Path(workspace).expanduser()
    db_path = Path(store_path.replace("{agentId}", agent_id)).expanduser()
    return workspace_dir, db_path


def prepare_memory_root(workspace_dir: Path, relative_root: str) -> Path:
    root = workspace_dir / relative_root
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_memory_documents(
    workspace_dir: Path,
    documents: list[MemoryDocument],
) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for document in documents:
        target = workspace_dir / document.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document.content, encoding="utf-8")
        logs.append(
            {
                **document.to_dict(),
                "absolute_path": str(target),
                "bytes_written": target.stat().st_size,
            }
        )
    return logs


def reindex_memory(agent_id: str) -> str:
    try:
        return run_openclaw_command(["openclaw", "memory", "index", "--agent", agent_id, "--force"])
    except OpenClawCliError as exc:
        raise MemoryCoreError(str(exc)) from exc


def extract_indexed_memory_chunks(
    db_path: Path,
    documents: list[MemoryDocument],
) -> list[MemoryChunk]:
    if not db_path.exists():
        raise MemoryCoreError(f"OpenClaw memory index does not exist: {db_path}")

    documents_by_path = {
        document.relative_path.replace("\\", "/"): document for document in documents
    }
    query = """
        SELECT path, start_line, end_line, text, embedding
        FROM chunks
        WHERE source = 'memory'
        ORDER BY path, start_line, end_line
    """

    chunks: list[MemoryChunk] = []
    try:
        with sqlite3.connect(db_path) as connection:
            cursor = connection.execute(query)
            for path_value, start_line, end_line, text, embedding_value in cursor.fetchall():
                if not isinstance(path_value, str):
                    continue
                normalized_path = path_value.replace("\\", "/")
                document = documents_by_path.get(normalized_path)
                if document is None:
                    continue

                embedding = _parse_embedding_payload(embedding_value)
                chunks.append(
                    MemoryChunk(
                        sample_id=document.sample_id,
                        session_key=document.session_key,
                        session_index=document.session_index,
                        date_time=document.date_time,
                        relative_path=document.relative_path,
                        start_line=int(start_line or 0),
                        end_line=int(end_line or 0),
                        content=str(text or ""),
                        embedding=embedding,
                    )
                )
    except sqlite3.Error as exc:
        raise MemoryCoreError(f"Failed to read OpenClaw memory index chunks: {exc}") from exc

    return chunks


def _parse_embedding_payload(value: object) -> list[float]:
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise MemoryCoreError("OpenClaw memory chunk embedding was not valid JSON") from exc
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
    return []
