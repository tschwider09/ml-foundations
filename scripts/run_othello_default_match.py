from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_foundations.othello.game_runner import run_default_match


if __name__ == "__main__":
    run_default_match()
