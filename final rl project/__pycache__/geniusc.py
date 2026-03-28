import gymnasium as gym
import numpy as np
import ale_py
import cv2
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, Flatten, Add, Multiply, Concatenate
from tensorflow.keras.layers import MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformObservation
from tensorflow.keras import mixed_precision
import threading
import time
import os

# Configure mixed precision properly
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Environment and training parameters
ENV_NAME       = "ALE/SpaceInvaders-v5"
MEMORY_SIZE    = 200000  
BATCH_SIZE     = 32     # Reduced for stability
GAMMA          = 0.99
LEARNING_RATE  = 1e-4   # Reduced learning rate
EPSILON_START  = 1.0
EPSILON_MIN    = 0.01    
EPSILON_DECAY  = 0.995  
TARGET_UPDATE  = 1000    
EPISODES       = 300
STACK_SIZE     = 4
IMG_HEIGHT     = 84
IMG_WIDTH      = 84
UPDATE_FREQ    = 4       
WARMUP_STEPS   = 5000   
DEATH_PENALTY  = 5
SURVIVAL_BONUS = 0.01   # Reduced bonus

def build_neat_inspired_model(input_shape, n_actions):
    """
    NEAT-inspired architecture with evolved-like connectivity patterns
    Fixed version with proper layer connections and shapes
    """
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)
    
    # Primary convolutional pathway (traditional CNN backbone)
    conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation="relu", padding="valid", name="conv1")(inputs)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation="relu", padding="valid", name="conv2")(conv1)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="valid", name="conv3")(conv2)
    
    # Secondary pathway (smaller filters, different receptive field)
    conv1b = Conv2D(16, (4, 4), strides=(2, 2), activation="relu", padding="valid", name="conv1b")(inputs)
    conv2b = Conv2D(32, (4, 4), strides=(2, 2), activation="relu", padding="valid", name="conv2b")(conv1b)
    conv3b = Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="valid", name="conv3b")(conv2b)
    
    # Global context pathway (large receptive field)
    conv1c = Conv2D(8, (8, 8), strides=(4, 4), activation="relu", padding="valid", name="conv1c")(inputs)
    conv2c = Conv2D(16, (3, 3), strides=(1, 1), activation="relu", padding="valid", name="conv2c")(conv1c)
    
    # Flatten pathways
    flat1 = Flatten(name="flat1")(conv3)
    flat2 = Flatten(name="flat2")(conv3b)
    flat3 = Flatten(name="flat3")(conv2c)
    
    # Modular processing units (inspired by NEAT's modular evolution)
    # Module 1: Main feature processing
    dense1a = Dense(256, activation='relu', name="dense1a")(flat1)
    dense1b = Dense(128, activation='relu', name="dense1b")(dense1a)
    
    # Module 2: Auxiliary feature processing  
    dense2a = Dense(128, activation='relu', name="dense2a")(flat2)
    dense2b = Dense(64, activation='relu', name="dense2b")(dense2a)
    
    # Module 3: Context processing
    dense3a = Dense(64, activation='relu', name="dense3a")(flat3)
    dense3b = Dense(32, activation='relu', name="dense3b")(dense3a)
    
    # Cross-module connections (skip connections like NEAT evolves)
    # Properly sized cross connections
    cross_12 = Dense(64, activation='relu', name="cross_12")(dense1a)
    merged_2 = Add(name="merged_2")([dense2b, cross_12])
    
    cross_13 = Dense(32, activation='relu', name="cross_13")(dense1b)
    merged_3 = Add(name="merged_3")([dense3b, cross_13])
    
    # Final integration layer (multi-pathway fusion)
    # Standardize dimensions for concatenation
    resize_1 = Dense(128, activation='relu', name="resize_1")(dense1b)
    resize_2 = Dense(128, activation='relu', name="resize_2")(merged_2)
    resize_3 = Dense(128, activation='relu', name="resize_3")(merged_3)
    
    # Multi-pathway fusion
    concatenated = Concatenate(name="concatenated")([resize_1, resize_2, resize_3])
    
    # Final processing layers with residual connections
    fusion1 = Dense(256, activation='relu', name="fusion1")(concatenated)
    fusion2 = Dense(256, activation='relu', name="fusion2")(fusion1)
    
    # Residual connection
    residual = Add(name="residual")([fusion1, fusion2])
    
    # Output heads (Dueling DQN architecture)
    # Value head
    value_head = Dense(128, activation='relu', name="value_head")(residual)
    value_out = Dense(1, activation='linear', name="value_out", dtype='float32')(value_head)
    
    # Advantage head  
    advantage_head = Dense(128, activation='relu', name="advantage_head")(residual)
    advantage_out = Dense(n_actions, activation='linear', name="advantage_out", dtype='float32')(advantage_head)
    
    # Dueling DQN combination using functional approach
    advantage_mean = tf.reduce_mean(advantage_out, axis=1, keepdims=True)
    advantage_centered = tf.subtract(advantage_out, advantage_mean)
    
    # Broadcast value to match advantage shape - fix for dynamic shape
    value_broadcast = tf.keras.layers.RepeatVector(n_actions)(tf.keras.layers.Flatten()(value_out))
    value_reshaped = tf.keras.layers.Reshape((n_actions,))(value_broadcast)
    
    q_values = tf.keras.layers.Add(name="q_values")([value_reshaped, advantage_centered])
    
    model = Model(inputs=inputs, outputs=q_values, name="NEAT_Inspired_DQN")
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss='huber_loss'
    )
    
    return model

def build_alternative_evolved_model(input_shape, n_actions):
    """
    Alternative architecture inspired by evolved topologies
    Fixed version with proper attention mechanism
    """
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)
    
    # Multi-scale feature extraction
    # Scale 1: Fine details
    s1_conv1 = Conv2D(16, (3, 3), strides=(1, 1), activation="relu", padding="same")(inputs)
    s1_conv2 = Conv2D(32, (3, 3), strides=(2, 2), activation="relu", padding="valid")(s1_conv1)
    s1_conv3 = Conv2D(64, (3, 3), strides=(2, 2), activation="relu", padding="valid")(s1_conv2)
    
    # Scale 2: Medium features  
    s2_conv1 = Conv2D(32, (5, 5), strides=(2, 2), activation="relu", padding="valid")(inputs)
    s2_conv2 = Conv2D(64, (3, 3), strides=(2, 2), activation="relu", padding="valid")(s2_conv1)
    
    # Scale 3: Large features
    s3_conv1 = Conv2D(64, (8, 8), strides=(4, 4), activation="relu", padding="valid")(inputs)
    s3_conv2 = Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="valid")(s3_conv1)
    
    # Global average pooling for attention weights
    s1_gap = GlobalAveragePooling2D()(s1_conv3)
    s2_gap = GlobalAveragePooling2D()(s2_conv2)  
    s3_gap = GlobalAveragePooling2D()(s3_conv2)
    
    # Attention mechanism
    attention_input = Concatenate()([s1_gap, s2_gap, s3_gap])
    attention_weights = Dense(3, activation='softmax')(attention_input)
    
    # Apply attention to flattened features
    s1_flat = Flatten()(s1_conv3)
    s2_flat = Flatten()(s2_conv2)
    s3_flat = Flatten()(s3_conv2)
    
    # Weighted combination using proper broadcasting
    def apply_attention_weight(inputs):
        features, weights, scale_idx = inputs
        weight = tf.expand_dims(weights[:, scale_idx], axis=1)
        return features * weight
    
    weighted_s1 = tf.keras.layers.Lambda(lambda x: apply_attention_weight([x[0], x[1], 0]))([s1_flat, attention_weights])
    weighted_s2 = tf.keras.layers.Lambda(lambda x: apply_attention_weight([x[0], x[1], 1]))([s2_flat, attention_weights])
    weighted_s3 = tf.keras.layers.Lambda(lambda x: apply_attention_weight([x[0], x[1], 2]))([s3_flat, attention_weights])
    
    # Feature fusion with proper dimensionality matching
    # Resize all features to same dimension
    resize_s1 = Dense(256, activation='relu')(weighted_s1)
    resize_s2 = Dense(256, activation='relu')(weighted_s2)
    resize_s3 = Dense(256, activation='relu')(weighted_s3)
    
    combined = Add()([resize_s1, resize_s2, resize_s3])
    
    # Complex routing network
    route1 = Dense(512, activation='relu')(combined)
    route2 = Dense(256, activation='relu')(route1)
    route3 = Dense(256, activation='relu')(combined)  # Skip connection
    
    final_features = Add()([route2, route3])
    
    # Output layer
    outputs = Dense(n_actions, activation='linear', dtype='float32')(final_features)
    
    model = Model(inputs=inputs, outputs=outputs, name="Evolved_Architecture_DQN")
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss='huber_loss'
    )
    
    return model

def test_model(model, env_name, episodes=5):
    """Test the trained model"""
    print("TESTING TRAINED MODEL")
    test_env = gym.make(env_name, frameskip=1, render_mode='human')
    test_env = AtariPreprocessing(test_env, grayscale_obs=True, scale_obs=True, frame_skip=4)
    test_env = FrameStackObservation(test_env, stack_size=4)
    
    # Create new observation space for the transposed observations
    new_obs_space = gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(IMG_HEIGHT, IMG_WIDTH, STACK_SIZE),
        dtype=np.float32
    )
    
    test_env = TransformObservation(test_env, lambda obs: np.transpose(obs, (1, 2, 0)), new_obs_space)

    for episode in range(1, episodes+1):
        state, _ = test_env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Ensure state has correct shape and dtype
            state_input = np.expand_dims(state.astype(np.float32), axis=0)
            q_vals = model.predict(state_input, verbose=0)
            action = np.argmax(q_vals[0])
            next_state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            state = next_state

        print(f"Test Episode {episode:2d}: Score = {total_reward:6.1f}, Steps = {steps:4d}")

    test_env.close()

class PrioritizedReplayBuffer:
    """Fixed prioritized replay buffer with proper error handling"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None
            
        # Avoid division by zero
        valid_priorities = self.priorities[:len(self.buffer)]
        valid_priorities = np.maximum(valid_priorities, 1e-8)  # Prevent zeros
        
        probs = valid_priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(priority, 1e-8)  # Prevent zero priorities
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

def process_batch(batch, online_net, target_net, gamma):
    """Process batch with proper error handling and Double DQN"""
    states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
    
    # Ensure proper data types
    states = states.astype(np.float32)
    next_states = next_states.astype(np.float32)
    actions = actions.astype(np.int32)
    rewards = rewards.astype(np.float32)
    dones = dones.astype(np.bool_)
    
    # Double DQN: use online network to select actions, target network to evaluate
    q_next_online = online_net.predict(next_states, verbose=0)
    q_next_target = target_net.predict(next_states, verbose=0)
    
    # Select actions using online network
    next_actions = np.argmax(q_next_online, axis=1)
    
    # Get Q-values for current states
    q_current = online_net.predict(states, verbose=0)
    
    # Compute targets
    q_targets = q_current.copy()
    for i in range(len(batch)):
        if dones[i]:
            q_targets[i, actions[i]] = rewards[i]
        else:
            q_targets[i, actions[i]] = rewards[i] + gamma * q_next_target[i, next_actions[i]]
    
    return states, q_targets

def create_environment():
    """Create and configure the Atari environment"""
    env = gym.make(ENV_NAME, frameskip=1)
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)
    env = FrameStackObservation(env, stack_size=STACK_SIZE)
    
    # Create new observation space for the transposed observations
    old_obs_space = env.observation_space
    new_obs_space = gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(IMG_HEIGHT, IMG_WIDTH, STACK_SIZE),
        dtype=np.float32
    )
    
    env = TransformObservation(env, lambda obs: np.transpose(obs, (1, 2, 0)), new_obs_space)
    return env

# Main training loop with NEAT-inspired architecture
def train():
    """Main training function with improved error handling"""
    try:
        # Check GPU availability
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"GPU found: {len(physical_devices)} device(s)")
            # Configure GPU memory growth
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("No GPU found, using CPU")
    except Exception as e:
        print(f"GPU configuration warning: {e}")
    
    # Environment setup
    env = create_environment()
    n_actions = env.action_space.n
    input_shape = (IMG_HEIGHT, IMG_WIDTH, STACK_SIZE)

    # Initialize NEAT-inspired networks
    print("Building NEAT-inspired architecture...")
    online_net = build_neat_inspired_model(input_shape, n_actions)
    target_net = build_neat_inspired_model(input_shape, n_actions)
    target_net.set_weights(online_net.get_weights())
    
    # Print model architecture
    print(f"Model has {online_net.count_params():,} parameters")
    print("Architecture summary:")
    online_net.summary()

    memory = PrioritizedReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    step_count = 0
    best_score = -float('inf')
    scores_window = deque(maxlen=100)
    
    print("Starting training with NEAT-inspired architecture...")
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
                state_input = np.expand_dims(state.astype(np.float32), axis=0)
                q_vals = online_net.predict(state_input, verbose=0)
                action = np.argmax(q_vals[0])

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Add survival bonus and death penalty
            current_lives = info.get('lives', lives)
            if lives > current_lives:
                reward -= DEATH_PENALTY
            else:
                reward += SURVIVAL_BONUS
            lives = current_lives
            
            total_reward += reward
            
            # Reward clipping for stability
            clipped_reward = np.clip(reward, -10, 10)
            
            # Store transition
            memory.add(state, action, clipped_reward, next_state, done)
            state = next_state

            # Learning step
            if len(memory) >= WARMUP_STEPS and step_count % UPDATE_FREQ == 0:
                try:
                    batch_data = memory.sample(BATCH_SIZE)
                    if batch_data is not None:
                        batch, indices, weights = batch_data
                        states_mb, q_targets = process_batch(batch, online_net, target_net, GAMMA)
                        
                        # Train with importance sampling weights
                        loss = online_net.train_on_batch(states_mb, q_targets, sample_weight=weights)
                        
                        # Update priorities
                        q_predicted = online_net.predict(states_mb, verbose=0)
                        td_errors = np.abs(q_targets - q_predicted)
                        priorities = np.mean(td_errors, axis=1) + 1e-6
                        memory.update_priorities(indices, priorities)
                        
                except Exception as e:
                    print(f"Training step error: {e}")
                    continue

            # Update target network
            if step_count % TARGET_UPDATE == 0:
                target_net.set_weights(online_net.get_weights())
                print(f"Target network updated at step {step_count}")

        # Exponential epsilon decay
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        scores_window.append(total_reward)
        avg_score = np.mean(scores_window) if len(scores_window) > 0 else total_reward
        
        episode_time = time.time() - episode_start
        
        if episode % 10 == 0 or episode == 1:
            elapsed_time = time.time() - start_time
            print(f"Episode: {episode:4d} | Score: {total_reward:6.1f} | "
                  f"Avg: {avg_score:6.1f} | Epsilon: {epsilon:.4f} | "
                  f"Steps: {step_count:6d} | Memory: {len(memory):6d} | "
                  f"Time: {elapsed_time:.1f}s")
        
        # Update best score and save model
        if total_reward > best_score:
            best_score = total_reward
            if episode > 50:  # Only save after some training
                os.makedirs('models', exist_ok=True)
                online_net.save_weights('models/best_dqn_neat_weights.weights.h5')
        
        # Early stopping if performing well
        if len(scores_window) >= 100 and avg_score >= 200:
            print(f"Environment solved in {episode} episodes! Average score: {avg_score:.2f}")
            break
    
    env.close()
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")
    print(f"Best score achieved: {best_score:.1f}")
    
    # Save the final model
    os.makedirs('models', exist_ok=True)
    online_net.save('models/dqn_neat_inspired_final.keras')
    online_net.save_weights('models/dqn_neat_weights_final.weights.h5')

    # Test the trained model
    print("\nTesting final model...")
    test_model(online_net, ENV_NAME)
    return online_net

if __name__ == "__main__":
    model = train()