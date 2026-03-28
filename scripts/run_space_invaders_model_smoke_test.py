from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_foundations.atari_rl.model_smoke_test import main


if __name__ == "__main__":
    main()
