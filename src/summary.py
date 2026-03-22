from __future__ import annotations

from src.schema import JudgedResult, QaResult


def build_summary(
    qa_results: list[QaResult],
    judged_results: list[JudgedResult],
    *,
    run_label: str,
    input_path: str,
    limit: int | None,
) -> dict[str, object]:
    correct = sum(1 for item in judged_results if item.result == "CORRECT")
    wrong = sum(1 for item in judged_results if item.result == "WRONG")
    judged_count = len(judged_results)
    accuracy = correct / judged_count if judged_count else 0.0

    prompt_tokens = sum(item.token_usage.prompt_tokens or 0 for item in qa_results)
    completion_tokens = sum(item.token_usage.completion_tokens or 0 for item in qa_results)
    total_tokens = sum(item.token_usage.total_tokens or 0 for item in qa_results)

    latencies = [item.latency_seconds for item in qa_results if item.latency_seconds is not None]
    average_latency = sum(latencies) / len(latencies) if latencies else None

    return {
        "run_label": run_label,
        "input_path": input_path,
        "limit": limit,
        "selected_rows": len(qa_results),
        "judged_rows": judged_count,
        "correct": correct,
        "wrong": wrong,
        "task_completion_rate": accuracy,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "average_latency_seconds": average_latency,
    }
