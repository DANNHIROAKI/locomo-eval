from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import batched
from pathlib import Path
from typing import Any

from openai import BadRequestError, OpenAI

from src.openclaw_cli import OpenClawCliError, load_openclaw_json, run_openclaw_command
from src.schema import MemoryChunk, MemoryRecord

TABLE_NAME = "memories"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_IMPORTANCE = 0.7
DEFAULT_CATEGORY = "fact"
DEFAULT_SCOPE = "global"
REDACTED_SECRET_SENTINELS = {
    "__OPENCLAW_REDACTED__",
}
EMBED_TRUNCATE_INITIAL_CHARS = 16000
EMBED_TRUNCATE_MIN_CHARS = 1000


class MemoryLanceDbError(RuntimeError):
    """Raised when direct LanceDB ingest cannot be completed."""


@dataclass(frozen=True)
class LanceDbConfig:
    backend: str
    db_path: Path
    embedding_model: str
    embedding_api_key: str | None
    embedding_base_url: str | None
    embedding_dimensions: int | None
    embedding_task_passage: str | None
    embedding_normalized: bool | None


@dataclass(frozen=True)
class LanceDbStatus:
    backend: str
    db_path: Path
    table_name: str
    exists: bool
    row_count: int | None

    @property
    def raw(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "db_path": str(self.db_path),
            "table_name": self.table_name,
            "exists": self.exists,
            "row_count": self.row_count,
        }


def make_lancedb_config(
    backend: str,
    db_path: Path,
    *,
    template: LanceDbConfig | None = None,
) -> LanceDbConfig:
    return LanceDbConfig(
        backend=backend,
        db_path=db_path,
        embedding_model=template.embedding_model if template else DEFAULT_EMBEDDING_MODEL,
        embedding_api_key=template.embedding_api_key if template else None,
        embedding_base_url=template.embedding_base_url if template else None,
        embedding_dimensions=template.embedding_dimensions if template else None,
        embedding_task_passage=template.embedding_task_passage if template else None,
        embedding_normalized=template.embedding_normalized if template else None,
    )


def resolve_lancedb_config(backend: str = "memory-lancedb") -> LanceDbConfig:
    try:
        payload = load_openclaw_json(
            ["openclaw", "config", "get", f"plugins.entries.{backend}.config", "--json"]
        )
    except OpenClawCliError as exc:
        raise MemoryLanceDbError(str(exc)) from exc

    if not isinstance(payload, dict):
        raise MemoryLanceDbError(f"OpenClaw did not return a {backend} config object")

    embedding = payload.get("embedding")
    if not isinstance(embedding, dict):
        raise MemoryLanceDbError(f"{backend} embedding config is missing")

    db_path = str(payload.get("dbPath", "")).strip()
    if not db_path:
        raise MemoryLanceDbError(f"{backend} dbPath is missing from OpenClaw config")

    return LanceDbConfig(
        backend=backend,
        db_path=Path(db_path).expanduser(),
        embedding_model=str(embedding.get("model", DEFAULT_EMBEDDING_MODEL)),
        embedding_api_key=_resolve_api_key(embedding.get("apiKey")),
        embedding_base_url=_optional_string(embedding.get("baseURL"))
        or _optional_string(embedding.get("baseUrl")),
        embedding_dimensions=_optional_int(embedding.get("dimensions")),
        embedding_task_passage=_optional_string(embedding.get("taskPassage")),
        embedding_normalized=_optional_bool(embedding.get("normalized")),
    )


def resolve_lancedb_status(config: LanceDbConfig, *, row_count: int | None = None) -> LanceDbStatus:
    return LanceDbStatus(
        backend=config.backend,
        db_path=config.db_path,
        table_name=TABLE_NAME,
        exists=config.db_path.exists(),
        row_count=row_count,
    )


def prepare_lancedb_store(db_path: Path) -> None:
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)


def count_lancedb_rows(config: LanceDbConfig) -> int | None:
    return count_lancedb_rows_at_path(config.db_path)


def count_lancedb_rows_at_path(db_path: Path) -> int | None:
    if not db_path.exists():
        return None

    try:
        import lancedb
    except ImportError as exc:
        raise MemoryLanceDbError(
            "The Python 'lancedb' package is required for LanceDB benchmark runs. Run 'uv sync'."
        ) from exc

    db = lancedb.connect(str(db_path))
    table_names = set(db.table_names())
    if TABLE_NAME not in table_names:
        return 0
    table = db.open_table(TABLE_NAME)
    return int(table.count_rows())


def write_memory_records(
    config: LanceDbConfig,
    records: list[MemoryRecord],
) -> list[dict[str, Any]]:
    if not records:
        return []

    try:
        import lancedb
    except ImportError as exc:
        raise MemoryLanceDbError(
            "The Python 'lancedb' package is required for LanceDB benchmark runs. Run 'uv sync'."
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY") or config.embedding_api_key
    if not api_key:
        raise MemoryLanceDbError("OPENAI_API_KEY must be set for LanceDB embeddings")

    client = OpenAI(api_key=api_key, base_url=config.embedding_base_url)
    texts = [record.content for record in records]
    vectors = _embed_texts(
        client,
        texts,
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
        task_passage=config.embedding_task_passage,
        normalized=config.embedding_normalized,
    )

    if config.backend == "memory-lancedb":
        rows, ingest_log = _build_memory_lancedb_rows(config, records, vectors)
    elif config.backend == "memory-lancedb-pro":
        rows, ingest_log = _build_memory_lancedb_pro_rows(config, records, vectors)
    else:
        raise MemoryLanceDbError(f"Unsupported LanceDB backend: {config.backend}")

    db = lancedb.connect(str(config.db_path))
    table_names = set(db.table_names())
    if TABLE_NAME in table_names:
        db.drop_table(TABLE_NAME)
    db.create_table(TABLE_NAME, data=rows)
    return ingest_log


def write_memory_chunks(
    config: LanceDbConfig,
    chunks: list[MemoryChunk],
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    try:
        import lancedb
    except ImportError as exc:
        raise MemoryLanceDbError(
            "The Python 'lancedb' package is required for LanceDB benchmark runs. Run 'uv sync'."
        ) from exc

    if config.backend == "memory-lancedb":
        rows, ingest_log = _build_memory_lancedb_rows_from_chunks(config, chunks)
    elif config.backend == "memory-lancedb-pro":
        rows, ingest_log = _build_memory_lancedb_pro_rows_from_chunks(config, chunks)
    else:
        raise MemoryLanceDbError(f"Unsupported LanceDB backend: {config.backend}")

    db = lancedb.connect(str(config.db_path))
    table_names = set(db.table_names())
    if TABLE_NAME in table_names:
        db.drop_table(TABLE_NAME)
    db.create_table(TABLE_NAME, data=rows)
    return ingest_log


def write_memory_records_via_plugin_cli(
    config: LanceDbConfig,
    records: list[MemoryRecord],
    *,
    import_path: Path,
) -> list[dict[str, Any]]:
    if config.backend != "memory-lancedb-pro":
        raise MemoryLanceDbError(
            "Plugin CLI import is only supported for the memory-lancedb-pro backend"
        )

    import_payload = {
        "version": "1.0",
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "filters": {
            "scope": DEFAULT_SCOPE,
            "category": DEFAULT_CATEGORY,
        },
        "memories": [],
    }
    ingest_log: list[dict[str, Any]] = []
    for offset, record in enumerate(records):
        timestamp = _record_timestamp_ms(record, offset)
        metadata = _build_memory_lancedb_pro_metadata(record, timestamp)
        import_payload["memories"].append(
            {
                "id": str(uuid.uuid4()),
                "text": record.content,
                "category": DEFAULT_CATEGORY,
                "importance": DEFAULT_IMPORTANCE,
                "timestamp": timestamp,
                "metadata": metadata,
            }
        )
        ingest_log.append(
            {
                **record.to_dict(),
                "db_path": str(config.db_path),
                "table_name": TABLE_NAME,
                "scope": DEFAULT_SCOPE,
                "bytes_written": len(record.content.encode("utf-8")),
            }
        )

    import_path.parent.mkdir(parents=True, exist_ok=True)
    import_path.write_text(
        json.dumps(import_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    try:
        run_openclaw_command(["openclaw", "memory-pro", "delete-bulk", "--scope", DEFAULT_SCOPE])
        run_openclaw_command(
            ["openclaw", "memory-pro", "import", str(import_path), "--scope", DEFAULT_SCOPE]
        )
    except OpenClawCliError as exc:
        raise MemoryLanceDbError(f"memory-lancedb-pro CLI import failed: {exc}") from exc

    return ingest_log


def migrate_legacy_lancedb_to_pro(source_db: Path, *, scope: str = DEFAULT_SCOPE) -> str:
    try:
        run_openclaw_command(["openclaw", "memory-pro", "delete-bulk", "--scope", scope])
        return run_openclaw_command(
            [
                "openclaw",
                "memory-pro",
                "migrate",
                "run",
                "--source",
                str(source_db),
                "--default-scope",
                scope,
            ]
        )
    except OpenClawCliError as exc:
        raise MemoryLanceDbError(f"memory-lancedb-pro migration failed: {exc}") from exc


def _embed_texts(
    client: OpenAI,
    texts: list[str],
    *,
    model: str,
    dimensions: int | None,
    task_passage: str | None,
    normalized: bool | None,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for batch in batched(texts, 100):
        batch_inputs = list(batch)
        try:
            response = client.embeddings.create(
                **_embedding_params(
                    batch_inputs,
                    model=model,
                    dimensions=dimensions,
                    task_passage=task_passage,
                    normalized=normalized,
                )
            )
            embeddings.extend([list(item.embedding) for item in response.data])
        except BadRequestError as exc:
            if not _looks_like_context_limit_error(exc):
                raise
            for text in batch_inputs:
                embeddings.append(
                    _embed_single_text_resilient(
                        client,
                        text,
                        model=model,
                        dimensions=dimensions,
                        task_passage=task_passage,
                        normalized=normalized,
                    )
                )
    return embeddings


def _embed_single_text_resilient(
    client: OpenAI,
    text: str,
    *,
    model: str,
    dimensions: int | None,
    task_passage: str | None,
    normalized: bool | None,
) -> list[float]:
    candidate = text
    while True:
        try:
            response = client.embeddings.create(
                **_embedding_params(
                    candidate,
                    model=model,
                    dimensions=dimensions,
                    task_passage=task_passage,
                    normalized=normalized,
                )
            )
            return list(response.data[0].embedding)
        except BadRequestError as exc:
            if not _looks_like_context_limit_error(exc):
                raise
            next_candidate = _truncate_text_for_embedding(candidate)
            if next_candidate == candidate:
                raise MemoryLanceDbError(
                    "Embedding input exceeds provider limit even after truncation"
                ) from exc
            candidate = next_candidate


def _embedding_params(
    input_value: str | list[str],
    *,
    model: str,
    dimensions: int | None,
    task_passage: str | None,
    normalized: bool | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model,
        "input": input_value,
    }
    extra_body: dict[str, Any] = {}
    if dimensions is not None:
        params["dimensions"] = dimensions
    if task_passage is not None:
        extra_body["task"] = task_passage
    if normalized is not None:
        extra_body["normalized"] = normalized
    if extra_body:
        params["extra_body"] = extra_body
    return params


def _looks_like_context_limit_error(exc: BadRequestError) -> bool:
    message = str(exc).lower()
    return "maximum input length" in message or "context length" in message


def _truncate_text_for_embedding(text: str) -> str:
    normalized = text.strip()
    if len(normalized) <= EMBED_TRUNCATE_MIN_CHARS:
        return normalized

    if len(normalized) > EMBED_TRUNCATE_INITIAL_CHARS:
        target = EMBED_TRUNCATE_INITIAL_CHARS
    else:
        target = max(EMBED_TRUNCATE_MIN_CHARS, len(normalized) // 2)

    truncated = normalized[:target].rstrip()
    last_break = max(
        truncated.rfind("\n"),
        truncated.rfind(". "),
        truncated.rfind(" "),
    )
    if last_break >= EMBED_TRUNCATE_MIN_CHARS:
        truncated = truncated[:last_break].rstrip()
    return truncated


def _build_memory_lancedb_rows(
    config: LanceDbConfig,
    records: list[MemoryRecord],
    vectors: list[list[float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ingest_log: list[dict[str, Any]] = []
    for offset, (record, vector) in enumerate(zip(records, vectors, strict=True)):
        created_at = _record_timestamp_ms(record, offset)
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "text": record.content,
                "vector": vector,
                "importance": DEFAULT_IMPORTANCE,
                "category": DEFAULT_CATEGORY,
                "createdAt": created_at,
            }
        )
        ingest_log.append(
            {
                **record.to_dict(),
                "db_path": str(config.db_path),
                "table_name": TABLE_NAME,
                "bytes_written": len(record.content.encode("utf-8")),
            }
        )
    return rows, ingest_log


def _build_memory_lancedb_rows_from_chunks(
    config: LanceDbConfig,
    chunks: list[MemoryChunk],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ingest_log: list[dict[str, Any]] = []
    for offset, chunk in enumerate(chunks):
        created_at = _chunk_timestamp_ms(chunk, offset)
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "text": chunk.content,
                "vector": chunk.embedding,
                "importance": DEFAULT_IMPORTANCE,
                "category": DEFAULT_CATEGORY,
                "createdAt": created_at,
            }
        )
        ingest_log.append(
            {
                **chunk.to_dict(),
                "db_path": str(config.db_path),
                "table_name": TABLE_NAME,
                "bytes_written": len(chunk.content.encode("utf-8")),
            }
        )
    return rows, ingest_log


def _build_memory_lancedb_pro_rows(
    config: LanceDbConfig,
    records: list[MemoryRecord],
    vectors: list[list[float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ingest_log: list[dict[str, Any]] = []
    for offset, (record, vector) in enumerate(zip(records, vectors, strict=True)):
        timestamp = _record_timestamp_ms(record, offset)
        metadata = _build_memory_lancedb_pro_metadata(record, timestamp)
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "text": record.content,
                "vector": vector,
                "category": DEFAULT_CATEGORY,
                "scope": DEFAULT_SCOPE,
                "importance": DEFAULT_IMPORTANCE,
                "timestamp": timestamp,
                "metadata": metadata,
            }
        )
        ingest_log.append(
            {
                **record.to_dict(),
                "db_path": str(config.db_path),
                "table_name": TABLE_NAME,
                "scope": DEFAULT_SCOPE,
                "bytes_written": len(record.content.encode("utf-8")),
            }
        )
    return rows, ingest_log


def _build_memory_lancedb_pro_rows_from_chunks(
    config: LanceDbConfig,
    chunks: list[MemoryChunk],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ingest_log: list[dict[str, Any]] = []
    for offset, chunk in enumerate(chunks):
        timestamp = _chunk_timestamp_ms(chunk, offset)
        metadata = _build_memory_lancedb_pro_chunk_metadata(chunk, timestamp)
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "text": chunk.content,
                "vector": chunk.embedding,
                "category": DEFAULT_CATEGORY,
                "scope": DEFAULT_SCOPE,
                "importance": DEFAULT_IMPORTANCE,
                "timestamp": timestamp,
                "metadata": metadata,
            }
        )
        ingest_log.append(
            {
                **chunk.to_dict(),
                "db_path": str(config.db_path),
                "table_name": TABLE_NAME,
                "scope": DEFAULT_SCOPE,
                "bytes_written": len(chunk.content.encode("utf-8")),
            }
        )
    return rows, ingest_log


def _build_memory_lancedb_pro_metadata(record: MemoryRecord, timestamp: int) -> str:
    return json.dumps(
        {
            "source_session": f"{record.sample_id}/{record.session_key}",
            "tier": "working",
            "confidence": DEFAULT_IMPORTANCE,
            "access_count": 0,
            "last_accessed_at": timestamp,
            "locomo": {
                "sample_id": record.sample_id,
                "session_key": record.session_key,
                "session_index": record.session_index,
                "message_index": record.message_index,
                "dia_id": record.dia_id,
                "speaker": record.speaker,
                "session_date_time": record.date_time,
            },
        },
        ensure_ascii=False,
    )


def _build_memory_lancedb_pro_chunk_metadata(chunk: MemoryChunk, timestamp: int) -> str:
    return json.dumps(
        {
            "source_session": f"{chunk.sample_id}/{chunk.session_key}",
            "tier": "working",
            "confidence": DEFAULT_IMPORTANCE,
            "access_count": 0,
            "last_accessed_at": timestamp,
            "locomo": {
                "sample_id": chunk.sample_id,
                "session_key": chunk.session_key,
                "session_index": chunk.session_index,
                "session_date_time": chunk.date_time,
                "relative_path": chunk.relative_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            },
        },
        ensure_ascii=False,
    )


def _record_timestamp_ms(record: MemoryRecord, offset: int) -> int:
    parsed = _parse_record_datetime(record.date_time)
    if parsed is None:
        return int(time.time() * 1000) + offset
    return parsed + offset


def _chunk_timestamp_ms(chunk: MemoryChunk, offset: int) -> int:
    parsed = _parse_record_datetime(chunk.date_time)
    if parsed is None:
        return int(time.time() * 1000) + offset
    return parsed + offset


def _parse_record_datetime(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None

    cleaned = re.sub(r"\s+", " ", text).strip()
    formats = (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
        "%H:%M on %d %B, %Y",
        "%H:%M on %d %b, %Y",
    )
    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        return int(dt.timestamp() * 1000)
    return None


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _resolve_api_key(value: object) -> str | None:
    if isinstance(value, str):
        return _resolve_env_string(value)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                resolved = _resolve_env_string(item)
                if resolved is not None:
                    return resolved
    return None


def _resolve_env_string(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    if text in REDACTED_SECRET_SENTINELS:
        return None

    def replace_env_var(match: re.Match[str]) -> str:
        env_name = match.group(1)
        env_value = os.getenv(env_name)
        if env_value is None:
            raise MemoryLanceDbError(f"Environment variable {env_name} is not set")
        return env_value

    return re.sub(r"\$\{([^}]+)\}", replace_env_var, text)
