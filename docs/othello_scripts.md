# Othello Scripts and Modules

## Entry scripts
- `scripts/run_othello_default_match.py`
  - Runs the default AI-vs-AI Othello match.
- `scripts/run_othello_human_vs_baseline.py`
  - Runs human input versus baseline minimax AI.

## Core modules
- `ml_coursework/othello/board.py`
  - Board state, legal move logic, scoring, rendering helpers.
- `ml_coursework/othello/game_runner.py`
  - Main game loop, legality checks, time management.
- `ml_coursework/othello/human_player.py`
  - Human CLI move interface (`getMove`).

## Agents
- `ml_coursework/othello/agents/minimax_baseline.py`
  - Baseline heuristic minimax + alpha-beta pruning.
- `ml_coursework/othello/agents/minimax_numba.py`
  - Numba-accelerated minimax variant.
- `ml_coursework/othello/agents/minimax_numba_optimized.py`
  - Further optimized numba minimax variant.
- `ml_coursework/othello/agents/mcts_classic.py`
  - Classic Monte Carlo Tree Search agent.
- `ml_coursework/othello/agents/mcts_numba_rollout.py`
  - MCTS with numba rollout acceleration.
