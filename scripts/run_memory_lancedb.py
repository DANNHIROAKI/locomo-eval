from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runner import run_cli


if __name__ == "__main__":
    run_cli("memory-lancedb")
