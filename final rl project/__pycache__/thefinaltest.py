import gymnasium as gym
import numpy as np
import ale_py
import cv2
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformObservation
from tensorflow.keras import mixed_precision
import threading
import time
import os

# Enable mixed precision for faster training
mixed_precision.set_global_policy('mixed_float16')

# Optimized hyperparameters
ENV_NAME = "ALE/SpaceInvaders-v5"
MEMORY_SIZE = 100000  # Reduced for faster sampling
BATCH_SIZE = 128      # Increased for better GPU utilization
GAMMA = 0.99
LEARNING_RATE = 1e-4  # Reduced for stability
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995  # Slower decay
TARGET_UPDATE = 2000    # Less frequent updates
EPISODES = 200
STACK_SIZE = 4
IMG_HEIGHT = 84
IMG_WIDTH = 84
UPDATE_FREQ = 4
WARMUP_STEPS = 10000
DEATH_PENALTY = 10
SURVIVAL_BONUS = 0.01

def build_optimized_model(input_shape, n_actions):
    """Optimized model architecture for better performance"""
    model = Sequential([
        InputLayer(shape=input_shape),
        # More efficient convolution layers
        Conv2D(32, (8, 8), strides=(4, 4), activation="relu", padding="valid"),
        Conv2D(64, (4, 4), strides=(2, 2), activation="relu", padding="valid"),
        Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="valid"),
        Flatten(),
        # Smaller dense layers for faster computation
        Dense(256, activation='relu'),
        Dense(n_actions, activation='linear', dtype='float32')
    ])
    
    # Use AdamW optimizer for better convergence
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=1e-4,
        clipnorm=1.0
    )
    
    model.compile(optimizer=optimizer, loss='huber')
    return model

class FastReplayBuffer:
    """Optimized replay buffer using numpy arrays for faster access"""
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        
        # Pre-allocate arrays for faster access
        self.states = np.zeros((capacity,) + state_shape, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity,) + state_shape, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
    def add(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        if self.size < batch_size:
            return None
            
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size

@tf.function
def train_step(online_net, target_net, states, actions, rewards, next_states, dones, gamma):
    """Compiled training step for faster execution"""
    with tf.GradientTape() as tape:
        # Current Q-values
        current_q = online_net(states, training=True)
        current_q_action = tf.gather_nd(current_q, tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1))
        
        # Double DQN: use online network to select actions
        next_q_online = online_net(next_states, training=False)
        next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)
        
        # Use target network to evaluate selected actions
        next_q_target = target_net(next_states, training=False)
        next_q_action = tf.gather_nd(next_q_target, tf.stack([tf.range(tf.shape(next_actions)[0]), next_actions], axis=1))
        
        # Compute target Q-values
        target_q = rewards + gamma * next_q_action * (1.0 - tf.cast(dones, tf.float32))
        
        # Huber loss
        loss = tf.keras.losses.huber(target_q, current_q_action)
    
    # Apply gradients
    gradients = tape.gradient(loss, online_net.trainable_variables)
    online_net.optimizer.apply_gradients(zip(gradients, online_net.trainable_variables))
    
    return loss

def create_vectorized_env(num_envs=1):
    """Create vectorized environment for parallel data collection"""
    def make_env():
        env = gym.make(ENV_NAME, frameskip=1)
        env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)
        env = FrameStackObservation(env, stack_size=4)
        env = TransformObservation(env, lambda obs: np.transpose(obs, (1, 2, 0)), env.observation_space)
        return env
    
    if num_envs == 1:
        return make_env()
    else:
        return gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])

def test_model(model, env_name, episodes=5):
    """Test the trained model"""
    print("TESTING TRAINED MODEL")
    test_env = create_vectorized_env(1)
    
    for episode in range(1, episodes + 1):
        state, _ = test_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            q_vals = model(state[np.newaxis, :], training=False)
            action = tf.argmax(q_vals[0]).numpy()
            next_state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            state = next_state
        
        print(f"Test Episode {episode:2d}: Score = {total_reward:6.1f}, Steps = {steps:4d}")

def train():
    """Optimized training loop"""
    # GPU configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    
    # Environment setup
    env = create_vectorized_env(1)
    n_actions = env.action_space.n
    input_shape = (IMG_HEIGHT, IMG_WIDTH, STACK_SIZE)
    
    # Initialize networks
    online_net = build_optimized_model(input_shape, n_actions)
    target_net = build_optimized_model(input_shape, n_actions)
    target_net.set_weights(online_net.get_weights())
    
    # Initialize replay buffer
    memory = FastReplayBuffer(MEMORY_SIZE, input_shape)
    
    # Training variables
    epsilon = EPSILON_START
    step_count = 0
    best_score = -float('inf')
    scores_window = deque(maxlen=100)
    
    print("Starting optimized training...")
    start_time = time.time()
    
    for episode in range(1, EPISODES + 1):
        state, info = env.reset()
        done = False
        total_reward = 0
        episode_start = time.time()
        lives = info.get('lives', 3)
        
        while not done:
            step_count += 1
            
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                q_vals = online_net(state[np.newaxis, :], training=False)
                action = tf.argmax(q_vals[0]).numpy()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Reward shaping
            reward += SURVIVAL_BONUS
            current_lives = info.get('lives', lives)
            if lives > current_lives:
                reward -= DEATH_PENALTY
            lives = current_lives
            
            total_reward += reward
            
            # Clip rewards for stability
            clipped_reward = np.clip(reward, -1, 1)
            
            # Store transition
            memory.add(state, action, clipped_reward, next_state, done)
            state = next_state
            
            # Training step - vectorized and compiled
            if len(memory) >= WARMUP_STEPS and step_count % UPDATE_FREQ == 0:
                batch = memory.sample(BATCH_SIZE)
                if batch is not None:
                    states, actions, rewards, next_states, dones = batch
                    
                    # Convert to tensors
                    states = tf.constant(states)
                    actions = tf.constant(actions)
                    rewards = tf.constant(rewards)
                    next_states = tf.constant(next_states)
                    dones = tf.constant(dones)
                    
                    # Perform training step
                    loss = train_step(online_net, target_net, states, actions, 
                                    rewards, next_states, dones, GAMMA)
            
            # Update target network
            if step_count % TARGET_UPDATE == 0:
                target_net.set_weights(online_net.get_weights())
        
        # Update epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        scores_window.append(total_reward)
        avg_score = np.mean(scores_window)
        
        # Progress reporting
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Episode: {episode:4d} | Score: {total_reward:6.1f} | "
                  f"Avg: {avg_score:6.1f} | Epsilon: {epsilon:.4f} | "
                  f"Steps: {step_count:6d} | Time: {elapsed_time:.1f}s | "
                  f"FPS: {step_count/elapsed_time:.1f}")
        
        # Save best model
        if total_reward > best_score:
            best_score = total_reward
            os.makedirs('models', exist_ok=True)
            online_net.save('models/dqn_space_invaders_best.keras')
        
        # Early stopping
        if avg_score >= 500:
            print(f"Solved in {episode} episodes! Average score: {avg_score:.2f}")
            break
    
    env.close()
    print(f"Training completed in {time.time() - start_time:.1f} seconds")
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    online_net.save('models/dqn_space_invaders_final.keras')
    
    # Test the model
    test_model(online_net, ENV_NAME)
    return online_net

if __name__ == "__main__":
    model = train()