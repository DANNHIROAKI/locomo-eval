from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.schema import BenchmarkRow, MemoryDocument, MemoryRecord


def load_locomo_samples(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected LOCOMO input to be a list of samples")
    return data


def build_sample_lookup(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        lookup[sample_id] = sample
    return lookup


def flatten_benchmark_rows(samples: list[dict[str, Any]]) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    row_index = 1
    for sample in samples:
        sample_id = str(sample["sample_id"])
        for qa_index, qa in enumerate(sample.get("qa", [])):
            answer = str(qa.get("answer", "")).strip()
            if not answer:
                continue
            category = str(qa.get("category", ""))
            evidence = [str(item) for item in qa.get("evidence", [])]
            rows.append(
                BenchmarkRow(
                    benchmark_id=f"row-{row_index:06d}",
                    sample_id=sample_id,
                    qa_index=qa_index,
                    question=str(qa["question"]),
                    answer=answer,
                    category=category,
                    evidence=evidence,
                )
            )
            row_index += 1
    return rows


def select_rows(rows: list[BenchmarkRow], limit: int | None) -> list[BenchmarkRow]:
    if limit is None:
        return list(rows)
    return list(rows[:limit])


def selected_sample_ids(rows: list[BenchmarkRow]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.sample_id in seen:
            continue
        ordered.append(row.sample_id)
        seen.add(row.sample_id)
    return ordered


def selected_samples(
    sample_lookup: dict[str, dict[str, Any]], rows: list[BenchmarkRow]
) -> list[dict[str, Any]]:
    return [sample_lookup[sample_id] for sample_id in selected_sample_ids(rows)]


def user_for_sample(sample_id: str) -> str:
    return f"locomo-{sample_id}"


def qa_session_key(sample_id: str, benchmark_id: str) -> str:
    return f"locomo-{sample_id}-qa-{benchmark_id}"


def qa_prompt(question: str, memory_backend: str) -> str:
    if memory_backend in {"memory-lancedb", "memory-lancedb-pro"}:
        return (
            "You are answering a benchmark question from long-term memory.\n"
            "You must use OpenClaw memory tools before answering.\n"
            "Process:\n"
            "1. Run memory_recall with the full question and limit=5.\n"
            "2. Run memory_recall again with limit=5 and shorter keyword queries built from names, activities, objects, places, and time hints in the question.\n"
            "3. Compare the retrieved memories and answer only from the memories that actually support the answer.\n"
            "4. If the question asks when something happened, use session_date_time together with relative phrases such as yesterday, last year, last month, or last Friday.\n"
            "5. If one query fails, try another wording before concluding the answer is missing.\n"
            "Answer style:\n"
            "- Give the shortest direct answer that is supported by memory.\n"
            "- Do not answer from prior knowledge or unsupported hints.\n"
            "- Do not explain your search process.\n"
            "- If memory truly does not contain the answer, say exactly: Memory does not contain the answer.\n\n"
            f"Question: {question}"
        )

    return (
        "You are answering a benchmark question from long-term memory.\n"
        "Use OpenClaw memory tools before answering.\n"
        "Process:\n"
        "1. Run memory_search with the full question.\n"
        "2. Run memory_search again with shorter keyword queries built from names, activities, objects, or places in the question.\n"
        "3. If memory_search returns a promising result, use memory_get on that file and line range before answering.\n"
        "4. Base the answer only on retrieved memory, not on guesses.\n"
        "5. If the question asks when something happened, use session_date_time together with relative phrases in the retrieved text such as yesterday, last year, last month, or last Friday.\n"
        "6. If one search path fails, try another wording before concluding the answer is missing.\n"
        "Answer style:\n"
        "- Give the shortest direct answer that is supported by memory.\n"
        "- Do not explain your search process.\n"
        "- If memory truly does not contain the answer, say exactly: Memory does not contain the answer.\n\n"
        f"Question: {question}"
    )


def ordered_session_keys(sample: dict[str, Any]) -> list[str]:
    conversation = sample["conversation"]
    session_keys = [
        key
        for key in conversation
        if key.startswith("session_") and not key.endswith("_date_time")
    ]
    return sorted(session_keys, key=lambda key: int(key.split("_")[1]))


def build_memory_documents(
    sample: dict[str, Any],
    *,
    memory_root: str = "memory/locomo",
) -> list[MemoryDocument]:
    sample_id = str(sample["sample_id"])
    conversation = sample["conversation"]
    speaker_a = str(conversation.get("speaker_a", "")).strip()
    speaker_b = str(conversation.get("speaker_b", "")).strip()
    documents: list[MemoryDocument] = []

    for session_index, session_key in enumerate(ordered_session_keys(sample), start=1):
        date_time = str(conversation.get(f"{session_key}_date_time", "")).strip()
        relative_path = f"{memory_root}/{sample_id}/{session_key}.md"
        documents.append(
            MemoryDocument(
                sample_id=sample_id,
                session_key=session_key,
                session_index=session_index,
                date_time=date_time,
                relative_path=relative_path,
                content=render_session_markdown(
                    sample_id=sample_id,
                    session_key=session_key,
                    session_index=session_index,
                    date_time=date_time,
                    speaker_a=speaker_a,
                    speaker_b=speaker_b,
                    messages=conversation.get(session_key, []),
                ),
            )
        )
    return documents


def build_memory_records(sample: dict[str, Any]) -> list[MemoryRecord]:
    sample_id = str(sample["sample_id"])
    conversation = sample["conversation"]
    speaker_a = str(conversation.get("speaker_a", "")).strip()
    speaker_b = str(conversation.get("speaker_b", "")).strip()
    records: list[MemoryRecord] = []

    for session_index, session_key in enumerate(ordered_session_keys(sample), start=1):
        date_time = str(conversation.get(f"{session_key}_date_time", "")).strip()
        messages = conversation.get(session_key, [])
        for message_index, message in enumerate(messages, start=1):
            records.append(
                MemoryRecord(
                    sample_id=sample_id,
                    session_key=session_key,
                    session_index=session_index,
                    date_time=date_time,
                    message_index=message_index,
                    dia_id=str(message.get("dia_id", "")).strip(),
                    speaker=str(message.get("speaker", "")).strip(),
                    content=render_message_memory_text(
                        sample_id=sample_id,
                        session_key=session_key,
                        session_index=session_index,
                        date_time=date_time,
                        speaker_a=speaker_a,
                        speaker_b=speaker_b,
                        message_index=message_index,
                        message=message,
                    ),
                )
            )
    return records


def render_session_markdown(
    *,
    sample_id: str,
    session_key: str,
    session_index: int,
    date_time: str,
    speaker_a: str,
    speaker_b: str,
    messages: list[dict[str, Any]],
) -> str:
    lines = [
        "# LOCOMO Memory",
        "",
        f"sample_id: {sample_id}",
        f"session_key: {session_key}",
        f"session_index: {session_index}",
        f"session_date_time: {date_time}",
    ]
    if speaker_a:
        lines.append(f"speaker_a: {speaker_a}")
    if speaker_b:
        lines.append(f"speaker_b: {speaker_b}")
    lines.append("")

    for message_index, message in enumerate(messages, start=1):
        _append_message_lines(lines, message_index, message)

    return "\n".join(lines).rstrip() + "\n"


def render_message_memory_text(
    *,
    sample_id: str,
    session_key: str,
    session_index: int,
    date_time: str,
    speaker_a: str,
    speaker_b: str,
    message_index: int,
    message: dict[str, Any],
) -> str:
    lines = [
        f"sample_id: {sample_id}",
        f"session_key: {session_key}",
        f"session_index: {session_index}",
        f"session_date_time: {date_time}",
    ]
    if speaker_a:
        lines.append(f"speaker_a: {speaker_a}")
    if speaker_b:
        lines.append(f"speaker_b: {speaker_b}")
    lines.append("")
    _append_message_lines(lines, message_index, message)
    return "\n".join(lines).rstrip()


def _append_message_lines(lines: list[str], message_index: int, message: dict[str, Any]) -> None:
    lines.append(f"## Message {message_index}")
    lines.append(f"dia_id: {str(message.get('dia_id', '')).strip()}")
    lines.append(f"speaker: {str(message.get('speaker', '')).strip()}")
    _append_block(lines, "text", str(message.get("text", "")))

    image_urls = message.get("img_url", [])
    if isinstance(image_urls, str):
        image_urls = [image_urls]
    if isinstance(image_urls, list):
        cleaned_urls = [str(url).strip() for url in image_urls if str(url).strip()]
    else:
        cleaned_urls = []
    if cleaned_urls:
        lines.append("img_url:")
        for url in cleaned_urls:
            lines.append(f"  - {url}")

    caption = str(message.get("blip_caption", "")).strip()
    if caption:
        _append_block(lines, "blip_caption", caption)
    lines.append("")


def _append_block(lines: list[str], label: str, value: str) -> None:
    lines.append(f"{label}:")
    if not value:
        lines.append("  ")
        return
    for line in value.splitlines() or [""]:
        lines.append(f"  {line}")
