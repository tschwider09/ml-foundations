# Othello Scripts and Modules

## Entry scripts
- `scripts/run_othello_default_match.py`
  - Runs the default AI-vs-AI Othello match.
- `scripts/run_othello_human_vs_baseline.py`
  - Runs human input versus baseline minimax AI.

## Core modules
- `ml_foundations/othello/board.py`
  - Board state, legal move logic, scoring, rendering helpers.
- `ml_foundations/othello/game_runner.py`
  - Main game loop, legality checks, time management.
- `ml_foundations/othello/human_player.py`
  - Human CLI move interface (`getMove`).

## Agents
- `ml_foundations/othello/agents/minimax_baseline.py`
  - Baseline minimax + alpha-beta pruning.
- `ml_foundations/othello/agents/minimax_numba.py`
  - Numba-accelerated minimax variant.
- `ml_foundations/othello/agents/minimax_numba_optimized.py`
  - Further optimized numba minimax variant.
- `ml_foundations/othello/agents/mcts_classic.py`
  - Classic Monte Carlo Tree Search agent.
- `ml_foundations/othello/agents/mcts_numba_rollout.py`
  - MCTS with numba rollout acceleration.
