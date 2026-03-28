from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from .environment import create_atari_env
from .evaluation import evaluate_greedy_policy
from .modeling import build_dqn_cnn, clone_target_network
from .replay import PrioritizedReplayBuffer
from .training_utils import (
    compute_double_dqn_targets,
    epsilon_greedy_action,
    transitions_to_arrays,
)


@dataclass
class DQNTrainConfig:
    env_name: str = "ALE/SpaceInvaders-v5"
    episodes: int = 100
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    target_update_steps: int = 1000
    update_frequency_steps: int = 4
    warmup_steps: int = 5000
    memory_size: int = 200_000
    batch_size: int = 64
    reward_clip: float = 1.0
    checkpoint_dir: Path = Path(__file__).resolve().parent / "models"
    save_versioned_artifacts: bool = True
    run_tag: str | None = None


def train_space_invaders_dqn(config: DQNTrainConfig = DQNTrainConfig()):
    """Train a Double-DQN agent for Space Invaders using prioritized replay."""
    env = create_atari_env(config.env_name, render_mode=None, frame_skip=4, stack_size=4)
    n_actions = env.action_space.n
    input_shape = env.observation_space.shape

    online_net = build_dqn_cnn(input_shape, n_actions, learning_rate=config.learning_rate)
    target_net = clone_target_network(online_net)

    replay = PrioritizedReplayBuffer(config.memory_size)
    epsilon = config.epsilon_start
    step_count = 0
    best_score = -float("inf")

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    run_tag = config.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    best_paths = [config.checkpoint_dir / "dqn_space_invaders_best.keras"]
    final_paths = [config.checkpoint_dir / "dqn_space_invaders.keras"]
    if config.save_versioned_artifacts:
        best_paths.append(config.checkpoint_dir / f"dqn_space_invaders_best_{run_tag}.keras")
        final_paths.append(config.checkpoint_dir / f"dqn_space_invaders_{run_tag}.keras")

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            step_count += 1
            action = epsilon_greedy_action(online_net, state, epsilon, n_actions)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            clipped_reward = float(np.clip(reward, -config.reward_clip, config.reward_clip))

            replay.add(state, action, clipped_reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(replay) >= config.warmup_steps and step_count % config.update_frequency_steps == 0:
                try:
                    samples, indices, weights = replay.sample(config.batch_size)
                except ValueError:
                    samples = None

                if samples is not None:
                    states, actions, rewards, next_states, dones = transitions_to_arrays(samples)
                    targets, td_errors = compute_double_dqn_targets(
                        online_net,
                        target_net,
                        states,
                        actions,
                        rewards,
                        next_states,
                        dones,
                        config.gamma,
                    )
                    online_net.train_on_batch(states, targets, sample_weight=weights)
                    replay.update_priorities(indices, td_errors + 1e-6)

            if step_count % config.target_update_steps == 0:
                target_net.set_weights(online_net.get_weights())

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

        if total_reward > best_score:
            best_score = total_reward
            for output_path in best_paths:
                online_net.save(output_path)

        if episode % 10 == 0 or episode == 1:
            elapsed = time.time() - start
            print(
                f"Episode {episode:4d} | Score {total_reward:6.1f} | "
                f"Best {best_score:6.1f} | Epsilon {epsilon:.4f} | Steps {step_count:7d} | "
                f"Time {elapsed:.1f}s"
            )

    for output_path in final_paths:
        online_net.save(output_path)

    print("\nSaved model artifacts:")
    for output_path in best_paths + final_paths:
        print(f" - {output_path}")

    env.close()
    return online_net


def evaluate_saved_model(model_path: str | Path, episodes: int = 5):
    """Load a saved model and evaluate greedy performance on Space Invaders."""
    env = create_atari_env("ALE/SpaceInvaders-v5", render_mode="human", frame_skip=4, stack_size=4)
    model = tf.keras.models.load_model(model_path)
    scores = evaluate_greedy_policy(model, env, episodes=episodes, sleep_s=0.03)
    env.close()
    return scores


def main():
    train_space_invaders_dqn()


if __name__ == "__main__":
    main()
