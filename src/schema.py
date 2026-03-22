from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkRow:
    benchmark_id: str
    sample_id: str
    qa_index: int
    question: str
    answer: str
    category: str
    evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryDocument:
    sample_id: str
    session_key: str
    session_index: int
    date_time: str
    relative_path: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryRecord:
    sample_id: str
    session_key: str
    session_index: int
    date_time: str
    message_index: int
    dia_id: str
    speaker: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryChunk:
    sample_id: str
    session_key: str
    session_index: int
    date_time: str
    relative_path: str
    start_line: int
    end_line: int
    content: str
    embedding: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QaResult:
    benchmark_id: str
    sample_id: str
    qa_index: int
    question: str
    answer: str
    category: str
    evidence: list[str]
    response: str | None
    latency_seconds: float | None
    token_usage: TokenUsage
    error: str | None
    user: str
    session_key: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["token_usage"] = self.token_usage.to_dict()
        return data


@dataclass(frozen=True)
class JudgedResult:
    benchmark_id: str
    sample_id: str
    category: str
    result: str
    reasoning: str
    question: str
    answer: str
    response: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
