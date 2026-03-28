from __future__ import annotations

import numpy as np

from .replay import Transition


def transitions_to_arrays(batch: list[Transition]):
    """Convert sampled transition objects into numpy arrays."""
    states = np.array([t.state for t in batch], dtype=np.float32)
    actions = np.array([t.action for t in batch], dtype=np.int32)
    rewards = np.array([t.reward for t in batch], dtype=np.float32)
    next_states = np.array([t.next_state for t in batch], dtype=np.float32)
    dones = np.array([t.done for t in batch], dtype=np.bool_)
    return states, actions, rewards, next_states, dones


def epsilon_greedy_action(model, state: np.ndarray, epsilon: float, n_actions: int):
    """Choose either a random action or the greedy Q-value action."""
    if np.random.rand() < epsilon:
        return int(np.random.randint(n_actions))
    q_values = model.predict(state[np.newaxis, :], verbose=0)
    return int(np.argmax(q_values[0]))


def compute_double_dqn_targets(
    online_net,
    target_net,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    gamma: float,
):
    """Compute Double DQN targets and TD-errors for prioritized replay updates."""
    q_next_online = online_net.predict(next_states, verbose=0)
    q_next_target = target_net.predict(next_states, verbose=0)
    next_actions = np.argmax(q_next_online, axis=1)

    q_current = online_net.predict(states, verbose=0)
    q_targets = q_current.copy()

    td_errors = np.zeros(len(states), dtype=np.float32)
    for i in range(len(states)):
        if dones[i]:
            target = rewards[i]
        else:
            target = rewards[i] + gamma * q_next_target[i, next_actions[i]]
        td_errors[i] = abs(target - q_current[i, actions[i]])
        q_targets[i, actions[i]] = target

    return q_targets, td_errors
