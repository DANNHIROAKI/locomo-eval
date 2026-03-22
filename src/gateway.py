from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from src.schema import TokenUsage


class GatewayError(RuntimeError):
    """Raised when the OpenClaw gateway request fails."""


@dataclass(frozen=True)
class GatewayResponse:
    text: str | None
    token_usage: TokenUsage
    latency_seconds: float
    raw_body: dict[str, Any]


@dataclass(frozen=True)
class GatewayClient:
    base_url: str
    token: str | None = None
    model: str = "openclaw"
    timeout_seconds: float = 300.0

    def send_message(
        self,
        *,
        user: str,
        session_key: str,
        message: str,
    ) -> GatewayResponse:
        headers = {"Content-Type": "application/json", "x-openclaw-session-key": session_key}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        payload = {
            "model": self.model,
            "input": message,
            "stream": False,
            "user": user,
        }

        start_time = time.monotonic()
        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/v1/responses",
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise GatewayError(str(exc)) from exc

        latency_seconds = time.monotonic() - start_time
        body = response.json()
        return GatewayResponse(
            text=extract_response_text(body),
            token_usage=extract_token_usage(body),
            latency_seconds=latency_seconds,
            raw_body=body,
        )


def extract_response_text(body: dict[str, Any]) -> str | None:
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    return text

    for item in body.get("output", []):
        text = item.get("text")
        if isinstance(text, str):
            return text
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                return text

    return None


def extract_token_usage(body: dict[str, Any]) -> TokenUsage:
    # OpenClaw-style responses may expose either input/output or prompt/completion keys.
    usage = body.get("usage", {})
    prompt_tokens = _coerce_int(usage.get("prompt_tokens"))
    if prompt_tokens is None:
        prompt_tokens = _coerce_int(usage.get("input_tokens"))

    completion_tokens = _coerce_int(usage.get("completion_tokens"))
    if completion_tokens is None:
        completion_tokens = _coerce_int(usage.get("output_tokens"))

    total_tokens = _coerce_int(usage.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
