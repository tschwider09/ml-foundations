import gymnasium as gym
import numpy as np
import ale_py
import cv2
from collections import deque
import random
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, Flatten, Input, Lambda
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Hyperparameters
ENV_NAME       = "ALE/TimePilot-v5"
MEMORY_SIZE    = 100000
BATCH_SIZE     = 32
GAMMA          = 0.99
LEARNING_RATE  = 2.5e-4
EPSILON_START  = 1.0
EPSILON_MIN    = 0.1
EPSILON_DECAY  = 0.99  # Decay per episode
TARGET_UPDATE  = 10000  # steps (not episodes)
EPISODES       = 10
STACK_SIZE     = 4
IMG_HEIGHT     = 84
IMG_WIDTH      = 84
DOUBLE_DQN     = True  # Use Double DQN
MODEL_PATH     = "time_pilot_model"
LOG_FREQ       = 100   # Log every N steps
RENDER         = False  # Render the game (set to False for faster training)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

# Preprocessing: grayscale, resize, normalize
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized

# Stack frames for temporal context
def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        # Clear the stacked frames and populate with the same frame
        stacked_frames = deque([frame] * STACK_SIZE, maxlen=STACK_SIZE)
    else:
        # Append the new frame to stacked_frames
        stacked_frames.append(frame)
    
    # Stack the frames into a single array (shape: height, width, stack_size)
    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

# Build Q-network - improved architecture
def build_model(input_shape, n_actions):
    # Standard DQN architecture
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation="relu"))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

# Calculate TD target - supports both regular DQN and Double DQN
def calculate_targets(batch, online_net, target_net):
    states, actions, rewards, next_states, dones = batch
    
    if DOUBLE_DQN:
        # Double DQN: Select actions using online network, evaluate using target network
        online_next_q_values = online_net.predict(next_states, verbose=0)
        best_actions = np.argmax(online_next_q_values, axis=1)
        target_next_q_values = target_net.predict(next_states, verbose=0)
        next_q_values = target_next_q_values[np.arange(BATCH_SIZE), best_actions]
    else:
        # Regular DQN: Both select and evaluate using target network
        target_next_q_values = target_net.predict(next_states, verbose=0)
        next_q_values = np.max(target_next_q_values, axis=1)
    
    # Calculate target Q-values
    target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
    
    # Create targets array (copy of predictions for the batch)
    targets = online_net.predict(states, verbose=0)
    
    # Update only the Q-values for the actions that were taken
    targets[np.arange(BATCH_SIZE), actions] = target_q_values
    
    return targets

# Plot training progress
def plot_results(episodes, scores, avg_scores, epsilons, filename="training_results.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot scores and average scores
    ax1.plot(episodes, scores, label='Score', alpha=0.6)
    ax1.plot(episodes, avg_scores, label='Average Score (last 100)', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Scores')
    ax1.legend()
    ax1.grid(True)
    
    # Plot epsilon
    ax2.plot(episodes, epsilons)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Exploration Rate (Epsilon)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Main training loop
def train():
    if RENDER:
        env = gym.make(ENV_NAME, render_mode="human")
    else:
        env = gym.make(ENV_NAME)
        
    n_actions = env.action_space.n
    input_shape = (IMG_HEIGHT, IMG_WIDTH, STACK_SIZE)

    # Initialize networks
    print("Building models...")
    online_net = build_model(input_shape, n_actions)
    target_net = build_model(input_shape, n_actions)
    target_net.set_weights(online_net.get_weights())

    # Initialize replay buffer
    memory = ReplayBuffer(MEMORY_SIZE)

    # Training tracking variables
    epsilon = EPSILON_START
    scores = []
    episode_scores = []
    episode_list = []
    epsilon_list = []
    avg_scores = []
    total_steps = 0
    start_time = time.time()
    
    # Create directory to save models
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    print("Starting training...")
    for episode in range(1, EPISODES+1):
        state, _ = env.reset()
        stacked_frames = deque(maxlen=STACK_SIZE)
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        done = False
        episode_reward = 0
        episode_start_time = time.time()
        step = 0

        while not done:
            step += 1
            total_steps += 1
            
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                q_vals = online_net.predict(state[np.newaxis, :], verbose=0)
                action = np.argmax(q_vals[0])

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Process and stack the next frame
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            # Store transition in replay buffer
            memory.add(state, action, reward, next_state, int(done))
            state = next_state

            # Train if we have enough samples in memory
            if memory.size() >= BATCH_SIZE:
                # Sample a batch from memory
                batch = memory.sample(BATCH_SIZE)
                states_mb = np.array([experience[0] for experience in batch])
                actions_mb = np.array([experience[1] for experience in batch])
                rewards_mb = np.array([experience[2] for experience in batch])
                next_states_mb = np.array([experience[3] for experience in batch])
                dones_mb = np.array([experience[4] for experience in batch])
                
                # Calculate targets using helper function
                batch_data = (states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb)
                targets = calculate_targets(batch_data, online_net, target_net)
                
                # Train the online network
                online_net.train_on_batch(states_mb, targets)
            
            # Update target network periodically
            if total_steps % TARGET_UPDATE == 0:
                target_net.set_weights(online_net.get_weights())
                print(f"Target network updated at step {total_steps}")
            
            # Log information periodically
            if total_steps % LOG_FREQ == 0:
                print(f"Step {total_steps}, Epsilon: {epsilon:.4f}, Memory: {memory.size()}")
        
        # Episode complete
        scores.append(episode_reward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)  # Decay epsilon
        
        # Calculate average score (last 100 episodes)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        
        # Track for plotting
        episode_list.append(episode)
        episode_scores.append(episode_reward)
        epsilon_list.append(epsilon)
        avg_scores.append(avg_score)
        
        # Print episode summary
        episode_time = time.time() - episode_start_time
        print(f"Episode: {episode}/{EPISODES}, Score: {episode_reward}, Avg Score: {avg_score:.2f}, Steps: {step}, Time: {episode_time:.2f}s, Epsilon: {epsilon:.4f}")
        
        # Save model periodically
        if episode % 50 == 0:
            model_file = os.path.join(MODEL_PATH, f"dqn_episode_{episode}.h5")
            online_net.save(model_file)
            print(f"Model saved to {model_file}")
            
            # Plot and save training progress
            plot_results(episode_list, episode_scores, avg_scores, epsilon_list)
    
    # Training complete
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Save final model
    final_model_file = os.path.join(MODEL_PATH, "dqn_final.h5")
    online_net.save(final_model_file)
    print(f"Final model saved to {final_model_file}")
    
    # Save final plot
    plot_results(episode_list, episode_scores, avg_scores, epsilon_list)
    
    env.close()
    return scores

# Test the trained model
def test(model_path, episodes=10):
    env = gym.make(ENV_NAME, render_mode="human")
    model = tf.keras.models.load_model(model_path)
    
    for episode in range(episodes):
        state, _ = env.reset()
        stacked_frames = deque(maxlen=STACK_SIZE)
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        done = False
        total_reward = 0
        
        while not done:
            # Get action from model
            q_vals = model.predict(state[np.newaxis, :], verbose=0)
            action = np.argmax(q_vals[0])
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Process next state
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
        
        print(f"Test Episode: {episode+1}, Score: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    # Train the agent
    print("Training mode")
    scores = train()
    
    # Test the trained agent
    print("\nTesting mode")
    test(os.path.join(MODEL_PATH, "dqn_final.h5"))