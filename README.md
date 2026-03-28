# ML Coursework

Portfolio repository showcasing practical AI implementation from my AI and Machine Learning Course across adversarial search and deep reinforcement learning.

## Quick Gloss
- Built and refactored two end-to-end AI projects into a production-style Python package.
- Demonstrates core AI skills: minimax + MCTS search, DQN training/evaluation, model benchmarking, and reproducible experimentation.
- Includes runnable scripts, smoke tests, and documentation for both fast review and deeper technical inspection.

## Projects

### 1) Othello AI (Adversarial Search)
What it is:
- An Othello engine with multiple AI agents for head-to-head comparison.

AI skills demonstrated:
- Minimax with alpha-beta pruning.
- Monte Carlo Tree Search (MCTS) variants.
- Search performance optimization with numba-accelerated logic.

Run:
```bash
python scripts/run_othello_default_match.py
python scripts/run_othello_human_vs_baseline.py
```

Deep dive:
- `docs/othello_scripts.md`

### 2) Space Invaders DQN (Deep RL)
What it is:
- A Double-DQN pipeline for `ALE/SpaceInvaders-v5` with training, smoke tests, model playback, and model ranking.

AI skills demonstrated:
- CNN-based Q-network for Atari frame stacks.
- Double-DQN target computation.
- Prioritized replay buffer.
- Controlled evaluation and seeded benchmarking for model comparison.
- Artifact management for best-model promotion and playback selection.

Run:
```bash
python scripts/run_space_invaders_train_dqn.py
python scripts/run_space_invaders_random_policy_smoke_test.py
python scripts/run_space_invaders_model_smoke_test.py
python scripts/run_space_invaders_model_benchmark.py --episodes 2 --max-steps 500 --seed 123
python scripts/run_space_invaders_trained_model.py --headless --episodes 1
```

Deep dive:
- `docs/space_invaders_dqn_scripts.md`
- `docs/legacy_rl_triage.md`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Model Artifacts

Space Invaders model artifacts are saved under:
- `ml_coursework/atari_rl/models/`

Training outputs:
- `dqn_space_invaders.keras`
- `dqn_space_invaders_best.keras`
- `dqn_space_invaders_YYYYMMDD_HHMMSS.keras`
- `dqn_space_invaders_best_YYYYMMDD_HHMMSS.keras`

Playback model resolution priority:
1. `--model /path/to/model.keras`
2. `SPACE_INVADERS_MODEL_PATH`
3. Benchmark-selected best model (`model_benchmark_results.json`)
4. Canonical fallback models in `models/`

## Repository Layout

```text
ml_coursework/
  othello/
  atari_rl/
scripts/
docs/
```

## Validation
- `python3 -m compileall ml_coursework scripts`
- RL model smoke test and benchmark scripts execute successfully.
- Othello default match runs end-to-end (headless backend supported).

## License
MIT (see `LICENSE`).
