# Legacy RL Triage (Keep vs Drop)

Goal: preserve useful engineering primitives from legacy experimentation scripts while excluding brittle or low-signal code.

## Kept (refactored into `ml_foundations/atari_rl`)

### Environment and preprocessing
- Source inspiration: `geniusc.py`, `currentbest.py`, `thefinaltest.py`
- Kept as: `environment.py:create_atari_env`
- Why kept: consistent Atari wrappers and channel-order conversion are required for all DQN variants.

### DQN model architecture
- Source inspiration: `currentbest.py`, `currentcurrent_best.py`, `vecot_supercharged.py`
- Kept as: `modeling.py:build_dqn_cnn`
- Why kept: this is the most standard and reproducible CNN baseline in the legacy set.

### Replay buffers
- Source inspiration: `claudegym.py`, `currentbest.py`, `thefinaltest.py`
- Kept as:
  - `replay.py:ReplayBuffer`
  - `replay.py:PrioritizedReplayBuffer`
  - `replay.py:FastReplayBuffer`
- Why kept: these abstractions are reusable and independent from specific training loops.

### Training math utilities
- Source inspiration: `currentbest.py`, `geniusc.py`
- Kept as:
  - `training_utils.py:transitions_to_arrays`
  - `training_utils.py:epsilon_greedy_action`
  - `training_utils.py:compute_double_dqn_targets`
- Why kept: these are core DQN mechanics and easy to test in isolation.

### Evaluation utility
- Source inspiration: multiple `test_model(...)` helpers
- Kept as: `evaluation.py:evaluate_greedy_policy`
- Why kept: common evaluation loop with no architecture lock-in.

## Dropped (intentionally not migrated)

### Monkey-patch compatibility hacks
- Dropped: `testing2.py`, `tryingtowork.py` monkey-patching `keras-rl` internals.
- Reason: brittle and version-fragile; high maintenance risk.

### Duplicate / conflicting training loops
- Dropped: multiple near-duplicate `train()` implementations across `AIgym.py`, `currentbest.py`, `currentcurrent_best.py`, `notdieprogram.py`.
- Reason: heavy duplication, mixed quality, and unclear canonical behavior.

### Over-complex experimental variants
- Dropped: vectorized-threaded prototype in `vecot_supercharged.py` and NEAT-inspired network in `geniusc.py`.
- Reason: high complexity with low confidence in reproducibility and debugability.

### Script-level side effects
- Dropped: modules that execute long training at import time.
- Reason: poor ergonomics for library usage and testing.

## Result
The new RL package preserves reusable functionality without carrying forward unstable experimentation scaffolding.

## Scope decision
- RL in this repository is now standardized on `ALE/SpaceInvaders-v5` (course final result).
- Non-Space-Invaders environment variants are intentionally not carried forward.
