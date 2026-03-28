from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_coursework.othello.game_runner import run_human_vs_baseline


if __name__ == "__main__":
    run_human_vs_baseline()
