# ML Foundations

Public portfolio repository for machine learning and search projects, refactored from coursework into a cleaner engineering codebase.

## Repository Structure

```text
ml_foundations/
  othello/
    board.py
    game_runner.py
    human_player.py
    agents/
      minimax_baseline.py
      minimax_numba.py
      minimax_numba_optimized.py
      mcts_classic.py
      mcts_numba_rollout.py
  atari_rl/
    benchmark_models.py
    environment.py
    modeling.py
    model_smoke_test.py
    replay.py
    saved_models.py
    training_utils.py
    evaluation.py
    space_invaders_dqn.py
    random_policy_smoke_test.py
    play_trained_model.py
scripts/
  run_othello_default_match.py
  run_othello_human_vs_baseline.py
  run_rl_train_space_invaders_dqn.py
  run_rl_random_policy_smoke_test.py
  run_rl_trained_model.py
  run_space_invaders_model_smoke_test.py
  run_rl_model_smoke_test.py
  run_space_invaders_model_benchmark.py
  run_rl_model_benchmark.py
  clean_generated_files.sh
```

## What This Demonstrates
- Adversarial search: minimax with alpha-beta pruning and MCTS variants.
- Performance experimentation: numba-accelerated move generation and evaluation.
- RL environment integration: Gymnasium + ALE Atari workflows (Space Invaders only).
- Engineering quality: descriptive module naming, cleaner package structure, import-safe entrypoints, and repository hygiene.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Othello

### Othello default AI matchup
```bash
python scripts/run_othello_default_match.py
```

### Othello human vs baseline AI
```bash
python scripts/run_othello_human_vs_baseline.py
```

Detailed Othello script/module map:
- `docs/othello_scripts.md`

## Space Invaders DQN

### Train Space Invaders DQN
```bash
python scripts/run_space_invaders_train_dqn.py
```

### Train Space Invaders DQN (alias)
```bash
python scripts/run_rl_train_space_invaders_dqn.py
```

### Space Invaders random-policy smoke test
```bash
python scripts/run_space_invaders_random_policy_smoke_test.py
```

### Space Invaders random-policy smoke test (alias)
```bash
python scripts/run_rl_random_policy_smoke_test.py
```

### Space Invaders trained-model playback
```bash
python scripts/run_space_invaders_trained_model.py --headless --episodes 1
```

### Space Invaders trained-model playback (alias)
```bash
python scripts/run_rl_trained_model.py
```

### Use your own trained model
```bash
python scripts/run_space_invaders_trained_model.py --model /absolute/path/to/your_model.keras --episodes 5
```
or
```bash
SPACE_INVADERS_MODEL_PATH=/absolute/path/to/your_model.keras python scripts/run_space_invaders_trained_model.py
```

### Smoke-test all saved model artifacts
```bash
python scripts/run_space_invaders_model_smoke_test.py
```
Add `--strict` if you want the command to fail on any bad artifact.

### Benchmark all saved models and promote the best
```bash
python scripts/run_space_invaders_model_benchmark.py --episodes 2 --max-steps 500
```
Use `--seed` to make ranking reproducible across runs.

Benchmark writes:
- `ml_foundations/atari_rl/models/model_benchmark_results.json`

After benchmark promotion, playback defaults to:
- `ml_foundations/atari_rl/models/dqn_space_invaders_best.keras`
- `ml_foundations/atari_rl/models/dqn_space_invaders.keras`

Training also writes run-stamped artifacts in the same folder:
- `dqn_space_invaders_YYYYMMDD_HHMMSS.keras`
- `dqn_space_invaders_best_YYYYMMDD_HHMMSS.keras`

Detailed Space Invaders script/module map:
- `docs/space_invaders_dqn_scripts.md`

### Remove local `__pycache__` and bytecode files
```bash
./scripts/clean_generated_files.sh
```

Expected model path for playback:
- `ml_foundations/atari_rl/models/dqn_space_invaders.keras`

## Validation Performed
- Compile check:
  - `python3 -m compileall ml_foundations scripts`
- Othello smoke checks:
  - baseline and numba agents return legal opening moves for black/white
- Import safety:
  - package modules import without auto-launching gameplay

## Notes
- Atari runtime requires proper ALE setup in your environment.
- Trained model playback requires a local exported `.keras` model file.
- RL modules intentionally target `ALE/SpaceInvaders-v5` only.
- Legacy RL triage notes are documented in `docs/legacy_rl_triage.md`.

## License
MIT (see `LICENSE`).
