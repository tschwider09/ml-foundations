import gymnasium as gym
import numpy as np
import ale_py
import tensorflow as tf
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformObservation
import time

# Environment constants
ENV_NAME = "ALE/SpaceInvaders-v5"

def create_environment():
    """Create the same environment setup as used in training"""
    env = gym.make(ENV_NAME, frameskip=1, render_mode='human')
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)
    env = FrameStackObservation(env, stack_size=4)
    env = TransformObservation(env, lambda obs: np.transpose(obs, (1, 2, 0)), env.observation_space)
    return env

def play_game(model, n_episodes=5):
    """Play the game using the loaded model"""
    env = create_environment()
    
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
            q_values = model.predict(state[np.newaxis, :], verbose=0)
            action = np.argmax(q_values[0])
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            state = next_state
            
            # Small delay to make it watchable
            time.sleep(0.03)
        
        scores.append(total_reward)
        print(f"Episode {episode}: Score = {total_reward:.1f}, Steps = {steps}")
    
    env.close()
    
    print(f"\n--- Results ---")
    print(f"Average score: {np.mean(scores):.2f}")
    print(f"Best score: {np.max(scores):.1f}")
    
    return scores

def main():
    print("Loading DQN model for Space Invaders...")
    
    # Load the full .keras model
    model_path = 'models/dqn_space_invaders.keras'
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully!")
    
    # Play the game
    scores = play_game(model, n_episodes=5)

if __name__ == "__main__":
    main()