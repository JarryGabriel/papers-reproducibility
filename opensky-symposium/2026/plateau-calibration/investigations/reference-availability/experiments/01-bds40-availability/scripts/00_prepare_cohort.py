"""Replay the complete input-channel preparation from the supplied CSV inputs."""
import sys
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / "src/opensky26_plateau/project_paths.py").is_file())
sys.path.insert(0, str(root / "src"))

from opensky26_plateau.pipeline.replay import main  # noqa: E402

if __name__ == "__main__":
    main()
