from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize benchmark result directories into a markdown table"
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing benchmark run subdirectories",
    )
    return parser.parse_args()


def load_summaries(outputs_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(outputs_dir.glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_label": payload.get("run_label", path.parent.name),
                "selected_rows": int(payload.get("selected_rows", 0) or 0),
                "correct": int(payload.get("correct", 0) or 0),
                "wrong": int(payload.get("wrong", 0) or 0),
                "average_latency_seconds": float(payload.get("average_latency_seconds", 0.0) or 0.0),
                "summary_path": str(path),
            }
        )
    return rows


def render_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Backend | Rows | Correct | Wrong | Avg latency (s) | Summary |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run_label} | {selected_rows} | {correct} | {wrong} | {latency:.2f} | `{summary_path}` |".format(
                run_label=row["run_label"],
                selected_rows=row["selected_rows"],
                correct=row["correct"],
                wrong=row["wrong"],
                latency=row["average_latency_seconds"],
                summary_path=row["summary_path"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    rows = load_summaries(outputs_dir)
    if not rows:
        raise SystemExit(f"No summary.json files found under {outputs_dir}")
    print(render_markdown(rows))


if __name__ == "__main__":
    main()
