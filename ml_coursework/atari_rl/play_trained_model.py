import numpy as np
import tensorflow as tf
import time
import argparse

from .environment import create_atari_env
from .saved_models import resolve_playback_model_path

# Environment constants
ENV_NAME = "ALE/SpaceInvaders-v5"

def create_environment(render_mode="human"):
    """Create the same environment setup as used in training"""
    return create_atari_env(ENV_NAME, render_mode=render_mode, frame_skip=4, stack_size=4)

def play_game(model, n_episodes=5, render_mode="human"):
    """Play the game using the loaded model"""
    env = create_environment(render_mode=render_mode)
    
    print(f"Playing {n_episodes} episodes...")
    scores = []
    for episode in range(1, n_episodes + 1):
        print(f"\n--- Episode {episode} ---")
        
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Use greedy policy (no exploration)
            q_values = model(state[np.newaxis, :], training=False).numpy()
            action = np.argmax(q_values[0])
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            state = next_state
            
            # Small delay to make it watchable
            if render_mode == "human":
                time.sleep(0.03)
        
        scores.append(total_reward)
        print(f"Episode {episode}: Score = {total_reward:.1f}, Steps = {steps}")
    
    env.close()
    
    print(f"\n--- Results ---")
    print(f"Average score: {np.mean(scores):.2f}")
    print(f"Best score: {np.max(scores):.1f}")
    
    return scores

def resolve_model_path(cli_path: str | None = None):
    """
    Resolve model path with robust fallbacks:
    --model, env var, benchmark best, then canonical defaults.
    """
    return resolve_playback_model_path(cli_path)

def main():
    parser = argparse.ArgumentParser(description="Play a trained Space Invaders DQN model.")
    parser.add_argument(
        "--model",
        dest="model_path",
        default=None,
        help="Optional path to a .keras model file.",
    )
    parser.add_argument(
        "--episodes",
        dest="episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable human rendering for faster/CLI runs.",
    )
    args = parser.parse_args()

    resolved_model_path, source = resolve_model_path(args.model_path)
    print("Loading DQN model for Space Invaders...")
    print(f"Model selection source: {source}")
    print(f"Model path: {resolved_model_path}")

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {resolved_model_path}. "
            "Pass --model <path>, set SPACE_INVADERS_MODEL_PATH, "
            "or run benchmark to promote a best model."
        )

    model = tf.keras.models.load_model(resolved_model_path)
    print("✓ Model loaded successfully!")
    
    # Play the game
    render_mode = None if args.headless else "human"
    scores = play_game(model, n_episodes=args.episodes, render_mode=render_mode)

if __name__ == "__main__":
    main()
