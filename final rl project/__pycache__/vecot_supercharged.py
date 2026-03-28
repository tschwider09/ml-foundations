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
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

mixed_precision.set_global_policy('mixed_float16')

ENV_NAME       = "ALE/SpaceInvaders-v5"
NUM_ENVS       = 8          # Number of parallel environments
MEMORY_SIZE    = 100000     # Increased for more environments
BATCH_SIZE     = 64         # Larger batch size for vectorized processing
NUM_BATCHES    = 4          # Number of parallel batches to process
GAMMA          = 0.99
LEARNING_RATE  = 2.5e-4  
EPSILON_START  = 1.0
EPSILON_MIN    = 0.01    
EPSILON_DECAY  = 0.995  
TARGET_UPDATE  = 1000    
EPISODES       = 1000
STACK_SIZE     = 4
IMG_HEIGHT     = 84
IMG_WIDTH      = 84
UPDATE_FREQ    = 4          # More frequent updates with vectorized envs
WARMUP_STEPS   = 10000      # More warmup steps for larger buffer

# Improved Q-network architecture
def build_model(input_shape, n_actions):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    
    # Improved CNN architecture inspired by DQN paper
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation="relu", padding="valid"))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu", padding="valid"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="valid"))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_actions, activation='linear', dtype='float32'))
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss='huber'
    )
    return model

# Vectorized Prioritized Experience Replay Buffer
class VectorizedPrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, num_envs=1):
        self.capacity = capacity
        self.alpha = alpha
        self.num_envs = num_envs
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
        self.lock = threading.Lock()  # Thread safety for parallel access
        
    def add_batch(self, states, actions, rewards, next_states, dones):
        """Add multiple experiences at once from vectorized environments"""
        with self.lock:
            for i in range(len(states)):
                # Ensure state shapes are consistent
                if states[i] is None or next_states[i] is None:
                    continue
                    
                if len(self.buffer) < self.capacity:
                    self.buffer.append(None)
                
                self.buffer[self.pos] = (states[i], actions[i], rewards[i], next_states[i], dones[i])
                self.priorities[self.pos] = self.max_priority
                self.pos = (self.pos + 1) % self.capacity
        
    def sample_multiple_batches(self, batch_size, num_batches, beta=0.4):
        """Sample multiple batches simultaneously for parallel processing"""
        if len(self.buffer) < batch_size:
            return None
            
        all_batches = []
        all_indices = []
        all_weights = []
        
        with self.lock:
            probs = self.priorities[:len(self.buffer)] ** self.alpha
            probs /= probs.sum()
            
            for _ in range(num_batches):
                # Ensure we have enough samples
                if len(self.buffer) < batch_size:
                    continue
                    
                indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
                samples = [self.buffer[idx] for idx in indices if self.buffer[idx] is not None]
                
                # Skip if not enough valid samples
                if len(samples) < batch_size // 2:
                    continue
                
                # Importance sampling weights
                weights = (len(self.buffer) * probs[indices[:len(samples)]]) ** (-beta)
                weights /= weights.max()
                
                all_batches.append(samples)
                all_indices.append(indices[:len(samples)])
                all_weights.append(weights)
        
        if len(all_batches) == 0:
            return None
            
        return all_batches, all_indices, all_weights
    
    def update_priorities_batch(self, all_indices, all_priorities):
        """Update priorities for multiple batches"""
        with self.lock:
            for indices, priorities in zip(all_indices, all_priorities):
                for idx, priority in zip(indices, priorities):
                    self.priorities[idx] = priority
                    self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

# Vectorized Environment Wrapper
class VectorizedAtariEnv:
    def __init__(self, env_name, num_envs):
        self.num_envs = num_envs
        self.envs = []
        
        # Create individual environments
        for i in range(num_envs):
            env = gym.make(env_name, frameskip=1, render_mode=None)
            env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)
            env = FrameStackObservation(env, stack_size=4)
            env = TransformObservation(env, lambda obs: np.transpose(obs, (1, 2, 0)), env.observation_space)
            self.envs.append(env)
        
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        
    def reset(self):
        """Reset all environments and return stacked states"""
        states = []
        for env in self.envs:
            state, _ = env.reset()
            states.append(state)
        return np.array(states)
    
    def step(self, actions):
        """Step all environments with given actions"""
        states = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
            # Auto-reset if done
            if done:
                state, _ = env.reset()
                states[i] = state
        
        return np.array(states), np.array(rewards), np.array(dones), infos
    
    def close(self):
        for env in self.envs:
            env.close()

# Vectorized batch processing with parallel execution
def process_multiple_batches(batches, online_net, target_net, gamma):
    """Process multiple batches in parallel"""
    def process_single_batch(batch):
        if len(batch) == 0:
            return None, None
            
        try:
            states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
            
            # Ensure consistent shapes
            if len(states.shape) != 4 or states.shape[0] == 0:
                return None, None
            
            # Double DQN with error handling
            q_next_online = online_net.predict(next_states, verbose=0)
            q_next_target = target_net.predict(next_states, verbose=0)
            
            next_actions = np.argmax(q_next_online, axis=1)
            q_current = online_net.predict(states, verbose=0)
            
            q_targets = q_current.copy()
            for i in range(len(batch)):
                if dones[i]:
                    q_targets[i, actions[i]] = rewards[i]
                else:
                    q_targets[i, actions[i]] = rewards[i] + gamma * q_next_target[i, next_actions[i]]
            
            return states, q_targets
        except Exception as e:
            print(f"Error processing batch: {e}")
            return None, None
    
    # Process batches sequentially to avoid tensor shape issues
    results = []
    for batch in batches:
        result = process_single_batch(batch)
        if result[0] is not None:
            results.append(result)
    
    return results

# Vectorized action selection
def select_actions_vectorized(states, online_net, epsilon, n_actions):
    """Select actions for all environments simultaneously"""
    batch_size = len(states)
    actions = np.zeros(batch_size, dtype=int)
    
    # Random actions for exploration
    random_mask = np.random.rand(batch_size) < epsilon
    actions[random_mask] = np.random.randint(0, n_actions, size=np.sum(random_mask))
    
    # Greedy actions for exploitation
    if not np.all(random_mask):
        greedy_mask = ~random_mask
        if np.any(greedy_mask):
            q_vals = online_net.predict(states[greedy_mask], verbose=0)
            actions[greedy_mask] = np.argmax(q_vals, axis=1)
    
    return actions

# Main vectorized training loop
def train_vectorized():
    # Configure GPU
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print("GPU found and configured for Metal")
        else:
            print("No GPU found, using CPU")
    except Exception as e:
        print(f"GPU configuration warning: {e}")
    
    # Vectorized environment setup
    vec_env = VectorizedAtariEnv(ENV_NAME, NUM_ENVS)
    n_actions = vec_env.action_space.n
    input_shape = (IMG_HEIGHT, IMG_WIDTH, STACK_SIZE)

    # Initialize networks
    online_net = build_model(input_shape, n_actions)
    target_net = build_model(input_shape, n_actions)
    target_net.set_weights(online_net.get_weights())

    # Vectorized replay buffer
    memory = VectorizedPrioritizedReplayBuffer(MEMORY_SIZE, num_envs=NUM_ENVS)
    
    epsilon = EPSILON_START
    step_count = 0
    best_avg_score = -float('inf')
    scores_window = deque(maxlen=100)
    episode_scores = [0] * NUM_ENVS
    
    print(f"Starting vectorized training with {NUM_ENVS} parallel environments...")
    start_time = time.time()

    # Initialize environments
    states = vec_env.reset()
    
    for episode in range(1, EPISODES + 1):
        total_steps_this_episode = 0
        
        while total_steps_this_episode < 1000:  # Max steps per episode cycle
            step_count += NUM_ENVS  # We take NUM_ENVS steps simultaneously
            total_steps_this_episode += NUM_ENVS
            
            # Vectorized action selection
            actions = select_actions_vectorized(states, online_net, epsilon, n_actions)
            
            # Step all environments
            next_states, rewards, dones, _ = vec_env.step(actions)
            
            # Clip rewards and update episode scores
            clipped_rewards = np.clip(rewards, -1, 1)
            episode_scores = [score + reward for score, reward in zip(episode_scores, rewards)]
            
            # Add experiences to buffer
            memory.add_batch(states, actions, clipped_rewards, next_states, dones)
            states = next_states
            
            # Handle episode completions
            for i, done in enumerate(dones):
                if done:
                    scores_window.append(episode_scores[i])
                    episode_scores[i] = 0
            
            # Vectorized learning step
            if len(memory) >= WARMUP_STEPS and step_count % UPDATE_FREQ == 0:
                try:
                    batch_data = memory.sample_multiple_batches(BATCH_SIZE, NUM_BATCHES)
                    if batch_data is not None:
                        batches, all_indices, all_weights = batch_data
                        
                        # Process multiple batches 
                        batch_results = process_multiple_batches(batches, online_net, target_net, GAMMA)
                        
                        # Train on all valid batches
                        if len(batch_results) > 0:
                            total_loss = 0
                            all_priorities = []
                            valid_indices = []
                            
                            for i, ((states_mb, q_targets), weights) in enumerate(zip(batch_results, all_weights[:len(batch_results)])):
                                if states_mb is not None and q_targets is not None:
                                    # Ensure weights match batch size
                                    if len(weights) != len(states_mb):
                                        weights = weights[:len(states_mb)]
                                    
                                    loss = online_net.train_on_batch(states_mb, q_targets, sample_weight=weights)
                                    total_loss += loss
                                    
                                    # Calculate TD errors for priority updates
                                    pred_q = online_net.predict(states_mb, verbose=0)
                                    td_errors = np.abs(q_targets - pred_q)
                                    priorities = np.mean(td_errors, axis=1) + 1e-6
                                    all_priorities.append(priorities)
                                    valid_indices.append(all_indices[i][:len(priorities)])
                            
                            # Update priorities for valid batches only
                            if len(all_priorities) > 0:
                                memory.update_priorities_batch(valid_indices, all_priorities)
                        
                except Exception as e:
                    print(f"Training step error: {e}")
                    continue

            # Update target network
            if step_count % TARGET_UPDATE == 0:
                target_net.set_weights(online_net.get_weights())

        # Decay epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        # Statistics
        if len(scores_window) > 0:
            avg_score = np.mean(scores_window)
            
            # Save best model
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                online_net.save_weights('best_vectorized_dqn.weights.h5')
            
            if episode % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Episode: {episode:4d} | Avg Score: {avg_score:6.1f} | "
                      f"Best: {best_avg_score:6.1f} | Epsilon: {epsilon:.4f} | "
                      f"Steps: {step_count:7d} | Buffer: {len(memory):6d} | "
                      f"Time: {elapsed_time:.1f}s")
            
            # Early stopping
            if len(scores_window) >= 100 and avg_score >= 200:
                print(f"Solved in {episode} episodes! Average score: {avg_score:.2f}")
                break
    
    vec_env.close()
    print(f"Vectorized training completed in {time.time() - start_time:.1f} seconds")
    return online_net

# Testing function for the trained model
def test_model(model_path='best_vectorized_dqn.weights.h5', episodes=5):
    """Test the trained model"""
    env = gym.make(ENV_NAME, frameskip=1, render_mode='human')
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)
    env = FrameStackObservation(env, stack_size=4)
    env = TransformObservation(env, lambda obs: np.transpose(obs, (1, 2, 0)), env.observation_space)
    
    # Load trained model
    input_shape = (IMG_HEIGHT, IMG_WIDTH, STACK_SIZE)
    n_actions = env.action_space.n
    model = build_model(input_shape, n_actions)
    
    try:
        model.load_weights(model_path)
        print(f"Loaded model from {model_path}")
    except:
        print(f"Could not load model from {model_path}")
        return
    
    scores = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            q_vals = model.predict(state[np.newaxis, :], verbose=0)
            action = np.argmax(q_vals[0])
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        scores.append(total_reward)
        print(f"Test Episode {episode + 1}: Score = {total_reward}")
    
    print(f"Average test score: {np.mean(scores):.2f}")
    env.close()

if __name__ == "__main__":
    # Train the vectorized model
    model = train_vectorized()
    
    # Uncomment to test the trained model
    # test_model()