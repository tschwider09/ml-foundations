# Space Invaders DQN Scripts and Modules

Scope: RL in this repo is standardized on `ALE/SpaceInvaders-v5`.

## Entry scripts
- `scripts/run_space_invaders_train_dqn.py`
  - Trains the Space Invaders Double-DQN model.
- `scripts/run_space_invaders_random_policy_smoke_test.py`
  - Random-action sanity check for env integration.
- `scripts/run_space_invaders_trained_model.py`
  - Loads and plays a trained model.
- `scripts/run_space_invaders_model_smoke_test.py`
  - Verifies each saved model artifact loads and produces Q-values (`--strict` optional).
- `scripts/run_space_invaders_model_benchmark.py`
  - Evaluates saved models, ranks performance, and promotes the best model (`--max-steps` and `--seed` supported).

Compatibility aliases (same behavior):
- `scripts/run_rl_train_space_invaders_dqn.py`
- `scripts/run_rl_random_policy_smoke_test.py`
- `scripts/run_rl_trained_model.py`
- `scripts/run_rl_model_smoke_test.py`
- `scripts/run_rl_model_benchmark.py`

## Core RL modules
- `ml_coursework/atari_rl/environment.py`
  - ALE registration + Atari preprocessing + frame stacking.
- `ml_coursework/atari_rl/modeling.py`
  - DQN CNN architecture and target-network cloning.
- `ml_coursework/atari_rl/replay.py`
  - ReplayBuffer, PrioritizedReplayBuffer, FastReplayBuffer.
- `ml_coursework/atari_rl/training_utils.py`
  - Epsilon-greedy and Double-DQN target computation.
- `ml_coursework/atari_rl/evaluation.py`
  - Greedy policy evaluation loop.
- `ml_coursework/atari_rl/space_invaders_dqn.py`
  - Main training/evaluation orchestration.
- `ml_coursework/atari_rl/play_trained_model.py`
  - Model loading + gameplay playback.
- `ml_coursework/atari_rl/model_smoke_test.py`
  - Artifact load/inference validation across saved models.
- `ml_coursework/atari_rl/benchmark_models.py`
  - Multi-model evaluation and best-model promotion.
- `ml_coursework/atari_rl/saved_models.py`
  - Shared model discovery, loading, and playback path resolution.

## Trained model path options
`play_trained_model.py` resolves model path in this order:
1. `--model /path/to/model.keras`
2. `SPACE_INVADERS_MODEL_PATH=/path/to/model.keras`
3. best model from benchmark results json (if present)
4. `ml_coursework/atari_rl/models/dqn_space_invaders_best.keras`
5. `ml_coursework/atari_rl/models/dqn_space_invaders.keras`
6. newest `.keras` model in the models directory

`space_invaders_dqn.py` training saves:
- canonical: `dqn_space_invaders.keras`, `dqn_space_invaders_best.keras`
- versioned per run: `dqn_space_invaders_YYYYMMDD_HHMMSS.keras`, `dqn_space_invaders_best_YYYYMMDD_HHMMSS.keras`
