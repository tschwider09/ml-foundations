from __future__ import annotations

import time

import numpy as np


def evaluate_greedy_policy(
    model,
    env,
    episodes: int = 5,
    sleep_s: float = 0.0,
    max_steps_per_episode: int | None = None,
    seed_base: int | None = None,
):
    """Run deterministic evaluation episodes and return rewards."""
    scores = []

    for episode_idx in range(episodes):
        if seed_base is None:
            state, _ = env.reset()
        else:
            state, _ = env.reset(seed=seed_base + episode_idx)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Direct forward pass is substantially faster than `model.predict` in tight loops.
            q_values = model(state[np.newaxis, :], training=False).numpy()
            action = int(np.argmax(q_values[0]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            steps += 1

            if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                done = True

            if sleep_s > 0:
                time.sleep(sleep_s)

        scores.append(total_reward)

    return scores
