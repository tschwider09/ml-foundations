import gymnasium as gym
import numpy as np
import ale_py
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from gymnasium.wrappers import AtariPreprocessing, TransformObservation
from tensorflow.keras import mixed_precision
import time

# Configure TensorFlow for Mac compatibility
try:
    # Configure GPU memory growth for compatibility
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        # Don't set memory growth on Metal GPU as it's not supported
        print("Metal GPU detected")
except Exception as e:
    print(f"GPU configuration: {e}")

# Disable problematic optimizations on Mac
# tf.config.optimizer.set_jit(True)  # Disable XLA - causes issues on Mac Metal
# Mixed precision can cause issues on Mac Metal, so we'll disable it
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# Optimized hyperparameters
ENV_NAME = "ALE/SpaceInvaders-v5"
MEMORY_SIZE = 50000
BATCH_SIZE = 64          # Increased for better GPU utilization
GAMMA = 0.99
LEARNING_RATE = 2.5e-4
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 500      # More frequent updates
EPISODES = 500
STACK_SIZE = 4
IMG_HEIGHT = 84
IMG_WIDTH = 84
UPDATE_FREQ = 4          # More frequent training
WARMUP_STEPS = 5000
RENDER_FREQ = 100        # Render every 100 episodes
LOG_FREQ = 10            # Log every 10 episodes

class FastFrameStack:
    """Optimized frame stacking with pre-allocated arrays"""
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
    
    def reset(self, frame):
        for _ in range(self.stack_size):
            self.frames.append(frame)
        return np.stack(self.frames, axis=-1)
    
    def add_frame(self, frame):
        self.frames.append(frame)
        return np.stack(self.frames, axis=-1)

class FastReplayBuffer:
    """Optimized replay buffer with pre-allocated arrays"""
    def __init__(self, capacity):
        self.capacity = capacity
        # Pre-allocate numpy arrays for faster access
        self.states = np.zeros((capacity, IMG_HEIGHT, IMG_WIDTH, STACK_SIZE), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, IMG_HEIGHT, IMG_WIDTH, STACK_SIZE), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=bool)
        self.size = 0
        self.pos = 0
    
    def add(self, state, action, reward, next_state, done):
        # Convert to uint8 to save memory
        self.states[self.pos] = (state * 255).astype(np.uint8)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = (next_state * 255).astype(np.uint8)
        self.dones[self.pos] = done
        
        if self.size < self.capacity:
            self.size += 1
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        if self.size < batch_size:
            return None
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Convert back to float32 and normalize
        states = self.states[indices].astype(np.float32) / 255.0
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices].astype(np.float32) / 255.0
        dones = self.dones[indices]
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.size

def build_optimized_model(input_shape, n_actions):
    """Optimized model architecture with functional API"""
    inputs = Input(shape=input_shape, dtype=tf.float32)
    
    # Efficient CNN architecture
    x = Conv2D(32, 8, strides=4, activation='relu', kernel_initializer='he_normal')(inputs)
    x = Conv2D(64, 4, strides=2, activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(64, 3, strides=1, activation='relu', kernel_initializer='he_normal')(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)  # Smaller dense layer
    outputs = Dense(n_actions, activation='linear', dtype='float32', 
                   kernel_initializer='he_normal')(x)  # Keep output in float32
    
    model = Model(inputs, outputs)
    
    # Compile with optimized settings
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss='huber',
        jit_compile=True  # XLA compilation for faster execution
    )
    return model

@tf.function
def train_step(online_net, target_net, states, actions, rewards, next_states, dones, gamma):
    """Optimized training step with tf.function for speed"""
    with tf.GradientTape() as tape:
        # Current Q-values
        q_current = online_net(states, training=True)
        
        # Double DQN: use online network to select actions, target network to evaluate
        q_next_online = online_net(next_states, training=False)
        q_next_target = target_net(next_states, training=False)
        
        # Select actions using online network
        next_actions = tf.argmax(q_next_online, axis=1, output_type=tf.int32)
        
        # Get Q-values for selected actions
        batch_indices = tf.range(tf.shape(next_states)[0])
        next_indices = tf.stack([batch_indices, next_actions], axis=1)
        q_next_max = tf.gather_nd(q_next_target, next_indices)
        
        # Compute target Q-values
        targets = rewards + gamma * q_next_max * (1.0 - tf.cast(dones, tf.float32))
        
        # Get current Q-values for taken actions
        current_indices = tf.stack([batch_indices, actions], axis=1)
        q_current_selected = tf.gather_nd(q_current, current_indices)
        
        # Compute loss
        loss = tf.keras.losses.huber(targets, q_current_selected)
    
    # Apply gradients
    gradients = tape.gradient(loss, online_net.trainable_variables)
    online_net.optimizer.apply_gradients(zip(gradients, online_net.trainable_variables))
    
    return loss

class DQNAgent:
    """Optimized DQN Agent with caching and batch processing"""
    def __init__(self, input_shape, n_actions):
        self.online_net = build_optimized_model(input_shape, n_actions)
        self.target_net = build_optimized_model(input_shape, n_actions)
        self.target_net.set_weights(self.online_net.get_weights())
        
        self.n_actions = n_actions
        self.input_shape = input_shape
    
    def get_action(self, state, epsilon):
        """Fast action selection"""
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        
        # Use model.__call__ instead of predict for single samples
        q_vals = self.online_net(state[np.newaxis, :], training=False)
        return int(tf.argmax(q_vals[0]))
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_net.set_weights(self.online_net.get_weights())
    
    def save_weights(self, filepath):
        """Save model weights"""
        self.online_net.save_weights(filepath)
    
    def load_weights(self, filepath):
        """Load model weights"""
        self.online_net.load_weights(filepath)
        self.target_net.set_weights(self.online_net.get_weights())

def train():
    """Optimized training loop"""
    # Configure GPU
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print("GPU found and configured")
        else:
            print("No GPU found, using CPU")
    except Exception as e:
        print(f"GPU configuration warning: {e}")
    
    # Environment setup - no rendering during training
    env = gym.make(ENV_NAME, frameskip=1, render_mode=None)
    env = AtariPreprocessing(env, 
                           grayscale_obs=True, 
                           scale_obs=True, 
                           frame_skip=4,
                           terminal_on_life_loss=True)  # Faster convergence
    
    # Manual frame stacking for better control
    frame_stack = FastFrameStack(STACK_SIZE)
    
    n_actions = env.action_space.n
    input_shape = (IMG_HEIGHT, IMG_WIDTH, STACK_SIZE)
    
    # Initialize agent and memory
    agent = DQNAgent(input_shape, n_actions)
    memory = FastReplayBuffer(MEMORY_SIZE)
    
    # Training variables
    epsilon = EPSILON_START
    step_count = 0
    best_score = -float('inf')
    scores_window = deque(maxlen=100)
    
    print("Starting optimized training...")
    start_time = time.time()
    
    for episode in range(1, EPISODES + 1):
        # Reset environment and frame stack
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        
        done = False
        total_reward = 0
        episode_start = time.time()
        
        while not done:
            step_count += 1
            
            # Get action
            action = agent.get_action(state, epsilon)
            
            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Update frame stack
            next_state = frame_stack.add_frame(next_obs)
            
            # Clip reward for stability
            clipped_reward = np.clip(reward, -1, 1)
            
            # Store transition
            memory.add(state, action, clipped_reward, next_state, done)
            state = next_state
            
            # Training step - vectorized and optimized
            if len(memory) >= WARMUP_STEPS and step_count % UPDATE_FREQ == 0:
                batch_data = memory.sample(BATCH_SIZE)
                if batch_data is not None:
                    states, actions, rewards, next_states, dones = batch_data
                    
                    # Convert to tensors
                    states = tf.constant(states)
                    actions = tf.constant(actions)
                    rewards = tf.constant(rewards)
                    next_states = tf.constant(next_states)
                    dones = tf.constant(dones)
                    
                    try:
                        loss = train_step(agent.online_net, agent.target_net, 
                                        states, actions, rewards, next_states, dones, GAMMA)
                    except Exception as e:
                        print(f"Training step error: {e}")
                        continue
            
            # Update target network
            if step_count % TARGET_UPDATE == 0:
                agent.update_target_network()
        
        # Update epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        # Record score
        scores_window.append(total_reward)
        avg_score = np.mean(scores_window)
        
        # Save best model
        if total_reward > best_score:
            best_score = total_reward
            agent.save_weights('best_dqn_optimized.weights.h5')
        
        # Logging - reduced frequency
        if episode % LOG_FREQ == 0:
            elapsed_time = time.time() - start_time
            episode_time = time.time() - episode_start
            steps_per_sec = step_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"Episode: {episode:4d} | Score: {total_reward:6.1f} | "
                  f"Avg: {avg_score:6.1f} | Epsilon: {epsilon:.4f} | "
                  f"Steps: {step_count:6d} | SPS: {steps_per_sec:.1f} | "
                  f"Time: {elapsed_time:.1f}s")
        
        # Render occasionally for monitoring
        if episode % RENDER_FREQ == 0:
            print(f"Rendering episode {episode}...")
            # Could add rendering logic here if needed
        
        # Early stopping
        if len(scores_window) >= 100 and avg_score >= 200:
            print(f"Environment solved in {episode} episodes! Average score: {avg_score:.2f}")
            break
    
    env.close()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")
    print(f"Average steps per second: {step_count / training_time:.1f}")
    
    return agent

def test_agent(agent, episodes=5):
    """Test the trained agent with rendering"""
    env = gym.make(ENV_NAME, frameskip=1, render_mode="human")
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)
    
    frame_stack = FastFrameStack(STACK_SIZE)
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        done = False
        total_reward = 0
        
        while not done:
            action = agent.get_action(state, epsilon=0.01)  # Almost greedy
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = frame_stack.add_frame(next_obs)
            
            time.sleep(0.02)  # Slow down for viewing
        
        print(f"Test Episode {episode + 1}: Score = {total_reward}")
    
    env.close()

if __name__ == "__main__":
    # Train the agent
    trained_agent = train()
    
    # Test the agent
    print("\nTesting trained agent...")
    test_agent(trained_agent, episodes=3)