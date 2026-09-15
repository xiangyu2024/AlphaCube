"""Thin wrapper: see `python -m alphacube.cli --help`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alphacube.cli import main  # noqa: E402

if __name__ == "__main__":
    sub = {"build_synthetic_cube": "synth", "research_factors": "research", "run_mining": "mine"}["research_factors"]
    sys.exit(main([sub] + sys.argv[1:]))
