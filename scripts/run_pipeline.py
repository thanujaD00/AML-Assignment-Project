"""Convenience entry point: `python scripts/run_pipeline.py` from the project root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dengue.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
