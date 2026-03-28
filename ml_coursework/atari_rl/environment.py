from __future__ import annotations

import os
from pathlib import Path
import tempfile

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "ml_coursework_mpl"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformObservation
from gymnasium import error as gym_error


def create_atari_env(
    env_name: str,
    *,
    render_mode: str | None = None,
    frame_skip: int = 4,
    stack_size: int = 4,
    grayscale_obs: bool = True,
    scale_obs: bool = True,
):
    """Create a consistently preprocessed Atari environment for DQN-style agents."""
    try:
        import ale_py
        gym.register_envs(ale_py)
    except ImportError as exc:
        raise RuntimeError(
            "Atari ALE package is missing. Install with: "
            "pip install 'gymnasium[atari,accept-rom-license]'"
        ) from exc

    try:
        env = gym.make(env_name, frameskip=1, render_mode=render_mode)
    except (gym_error.NamespaceNotFound, gym_error.NameNotFound) as exc:
        raise RuntimeError(
            "Atari environment is unavailable. Install Gymnasium Atari support "
            "with: pip install 'gymnasium[atari,accept-rom-license]'"
        ) from exc
    env = AtariPreprocessing(
        env,
        grayscale_obs=grayscale_obs,
        scale_obs=scale_obs,
        frame_skip=frame_skip,
    )
    env = FrameStackObservation(env, stack_size=stack_size)

    # Convert from (stack, H, W) to (H, W, stack), which is friendlier for Conv2D.
    old_space = env.observation_space
    if len(old_space.shape) != 3:
        raise ValueError(f"Expected 3D observation shape, got: {old_space.shape}")

    h, w = old_space.shape[-2], old_space.shape[-1]
    new_space = gym.spaces.Box(
        low=0.0,
        high=1.0 if scale_obs else 255.0,
        shape=(h, w, stack_size),
        dtype=np.float32,
    )

    env = TransformObservation(env, lambda obs: np.transpose(obs, (1, 2, 0)), new_space)
    return env
