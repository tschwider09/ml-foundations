from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Simple uniform replay buffer."""

    def __init__(self, capacity: int):
        self._buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self._buffer.append(Transition(state, int(action), float(reward), next_state, bool(done)))

    def sample(self, batch_size: int) -> list[Transition]:
        if len(self._buffer) < batch_size:
            raise ValueError("Not enough samples in replay buffer.")
        return random.sample(self._buffer, batch_size)

    def __len__(self):
        return len(self._buffer)


class PrioritizedReplayBuffer:
    """Prioritized replay with proportional sampling and IS weights."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: list[Transition | None] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        transition = Transition(state, int(action), float(reward), next_state, bool(done))
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples in prioritized replay buffer.")

        valid_priorities = self.priorities[: len(self.buffer)]
        valid_priorities = np.maximum(valid_priorities, 1e-8)

        probs = valid_priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            safe_priority = max(float(priority), 1e-8)
            self.priorities[idx] = safe_priority
            self.max_priority = max(self.max_priority, safe_priority)

    def __len__(self):
        return len(self.buffer)


class FastReplayBuffer:
    """Numpy-backed replay buffer optimized for batch sampling throughput."""

    def __init__(self, capacity: int, state_shape: tuple[int, ...]):
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.states = np.zeros((capacity,) + state_shape, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity,) + state_shape, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = bool(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.size < batch_size:
            raise ValueError("Not enough samples in fast replay buffer.")
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size
