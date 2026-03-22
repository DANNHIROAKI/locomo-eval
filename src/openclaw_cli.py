from __future__ import annotations

import json
import subprocess
from typing import Any


class OpenClawCliError(RuntimeError):
    """Raised when an OpenClaw CLI command fails or returns invalid JSON."""


def run_openclaw_command(args: list[str]) -> str:
    result = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise OpenClawCliError(detail)
    return result.stdout


def load_openclaw_json(args: list[str]) -> Any:
    stdout = run_openclaw_command(args)
    return extract_json_payload(stdout)


def resolve_memory_slot() -> str:
    payload = load_openclaw_json(["openclaw", "config", "get", "plugins.slots.memory", "--json"])
    if not isinstance(payload, str) or not payload.strip():
        raise OpenClawCliError("OpenClaw did not return an active memory slot")
    return payload


def set_config_value(path: str, value: str) -> None:
    run_openclaw_command(["openclaw", "config", "set", path, value])


def extract_json_payload(stdout: str) -> Any:
    decoder = json.JSONDecoder()
    starters = set('{["-0123456789tfn')
    for index, char in enumerate(stdout):
        if char not in starters:
            continue
        try:
            payload, _ = decoder.raw_decode(stdout[index:])
            return payload
        except json.JSONDecodeError:
            continue
    raise OpenClawCliError("OpenClaw did not return parseable JSON output")
