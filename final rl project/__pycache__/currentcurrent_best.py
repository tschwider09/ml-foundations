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


mixed_precision.set_global_policy('mixed_float16')
#note I am training on difficulty one not 0 maybe change back
ENV_NAME       = "ALE/SpaceInvaders-v5"
MEMORY_SIZE    = 200000 
BATCH_SIZE     = 64    
GAMMA          = 0.99
LEARNING_RATE  = 2.5e-4 
EPSILON_START  = 1.0
EPSILON_MIN    = 0.01   
EPSILON_DECAY  = 0.995 
TARGET_UPDATE  = 1000   
EPISODES       = 100
STACK_SIZE     = 4
IMG_HEIGHT     = 84
IMG_WIDTH      = 84
UPDATE_FREQ    = 4      
WARMUP_STEPS   = 5000  
DEATH_PENALTY  = 5
SURVIVAL_BONUS = .025


def build_model(input_shape, n_actions):
   model = Sequential()
   model.add(InputLayer(shape=input_shape))
   #consider reducing the size


   #this is recommended though.
   model.add(Conv2D(32, (8, 8), strides=(4, 4), activation="relu", padding="valid"))
   model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu", padding="valid"))
   model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="valid"))
  
   model.add(Flatten())
   model.add(Dense(512, activation='relu'))
   model.add(Dense(n_actions, activation='linear', dtype='float32')) 
  
   model.compile(
       optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0), #stability fix
       loss='huber'  # More stable than MSE for RL
   )
   return model
def test_model(model, env_name, episodes=5):
   print("TESTING TRAINED MODEL")
   test_env = gym.make(env_name, frameskip=1, render_mode='human')
   test_env = AtariPreprocessing(test_env, grayscale_obs=True, scale_obs=True, frame_skip=4)
   test_env = FrameStackObservation(test_env, stack_size=4)
   test_env = TransformObservation(test_env, lambda obs: np.transpose(obs, (1, 2, 0)), test_env.observation_space)




   for episode in range(1, episodes+1):
       state, _ = test_env.reset()
       done = False
       total_reward = 0
       steps = 0


       while not done:
           q_vals = model.predict(state[np.newaxis, :], verbose=0)
           action = np.argmax(q_vals[0])
           next_state, reward, terminated, truncated, _ = test_env.step(action)
           done = terminated or truncated
           total_reward += reward
           steps += 1
           state = next_state


       print(f"Test Episode {episode:2d}: Score = {total_reward:6.1f}, Steps = {steps:4d}")


#found this in tensorforce was recommend by gymnasium and google so I used it
class PrioritizedReplayBuffer:
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
          
       probs = self.priorities[:len(self.buffer)] ** self.alpha
       probs /= probs.sum()
      
       indices = np.random.choice(len(self.buffer), batch_size, p=probs)
       samples = [self.buffer[idx] for idx in indices]
      
       weights = (len(self.buffer) * probs[indices]) ** (-beta)
       weights /= weights.max()
      
       return samples, indices, weights
  
   def update_priorities(self, indices, priorities):
       for idx, priority in zip(indices, priorities):
           self.priorities[idx] = priority
           self.max_priority = max(self.max_priority, priority)
  
   def __len__(self):
       return len(self.buffer)


def process_batch(batch, online_net, target_net, gamma):
   states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
  
   # Double DQN: use online network to select actions, target network to evaluate
   q_next_online = online_net.predict(next_states, verbose=0) #verbose is 0 for my consoles sake
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


# Main training loop with optimizations
def train():
   #Thanks tf-metal for this code
   try:
       physical_devices = tf.config.list_physical_devices('GPU')
       if physical_devices:
           print("GPU found and configured for Metal")
       else:
           print("No GPU found, using CPU")
   except Exception as e:
       print(f"GPU configuration warning: {e}")
  
   # Environment setup from gym
   #env = gym.make(ENV_NAME, frameskip=1, difficulty=1) #note put difficulty up as a test
   env = gym.make(ENV_NAME, frameskip=1)
   env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)
   env = FrameStackObservation(env, stack_size=4) #maybe try a higher fram stack
   env = TransformObservation(env, lambda obs: np.transpose(obs, (1, 2, 0)), env.observation_space) #necessary for transforming output to fit the conv2d architechture




   n_actions = env.action_space.n
   input_shape = (IMG_HEIGHT, IMG_WIDTH, STACK_SIZE)


   # Initialize networks from gym
   online_net = build_model(input_shape, n_actions)
   #online_net = tf.keras.models.load_model('models/dqn_space_invaders.keras') #this is for picking up training where I left off
   target_net = build_model(input_shape, n_actions)
   target_net.set_weights(online_net.get_weights())


   memory = PrioritizedReplayBuffer(MEMORY_SIZE)
  
   epsilon = EPSILON_START
   step_count = 0
   best_score = -float('inf')
   scores_window = deque(maxlen=100)
  
   # Compile models for faster execution
   online_net.compile(optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0), loss='huber')
  
   print("Starting training...")
   start_time = time.time()


   for episode in range(1, EPISODES + 1):
       state, info = env.reset()
       done = False
       total_reward = 0
       episode_start = time.time()
       lives = info.get('lives', 3)


       while not done:
           step_count += 1
          
           # Epsilon-greedy with exponential decay
           if np.random.rand() < epsilon:
               action = env.action_space.sample()
           else:
               q_vals = online_net.predict(state[np.newaxis, :], verbose=0) #verbose is here becuase contantly printing progress got annoying
               action = np.argmax(q_vals[0])


           next_state, reward, terminated, truncated, info = env.step(action)
           done = terminated or truncated
           reward += SURVIVAL_BONUS #survival reward
          
           current_lives = info.get('lives', lives)
           if lives > current_lives:
               reward -= DEATH_PENALTY #death penalty
           lives = current_lives
          
           total_reward += reward
          
           # Reward clipping for stability
           clipped_reward = np.clip(reward, -1, 1)
          
           #️memory.add(state, action, reward, next_state, done)


           # Store transition
           memory.add(state, action, clipped_reward, next_state, done)
           state = next_state


           # Learning step - only after warmup and every UPDATE_FREQ steps
           if len(memory) >= WARMUP_STEPS and step_count % UPDATE_FREQ == 0:
               try:
                   batch_data = memory.sample(BATCH_SIZE)
                   if batch_data is not None:
                       batch, indices, weights = batch_data
                       states_mb, q_targets = process_batch(batch, online_net, target_net, GAMMA)
                      
                       # Train with importance sampling weights
                       loss = online_net.train_on_batch(states_mb, q_targets, sample_weight=weights)
                      
                       # Update priorities (simplified)
                       td_errors = np.abs(q_targets - online_net.predict(states_mb, verbose=0))
                       priorities = np.mean(td_errors, axis=1) + 1e-6 #this is what gym said worked best
                       memory.update_priorities(indices, priorities)
               except Exception as e:
                   print(f"Training step error: {e}")
                   continue


           # Update target network
           if step_count % TARGET_UPDATE == 0:
               target_net.set_weights(online_net.get_weights())


       # Exponential epsilon decay
       epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
      
       scores_window.append(total_reward)
       avg_score = np.mean(scores_window)
      
     
      
       episode_time = time.time() - episode_start
      
       if episode % 10 == 0:
           elapsed_time = time.time() - start_time
           print(f"Episode: {episode:4d} | Score: {total_reward:6.1f} | "
                 f"Avg: {avg_score:6.1f} | Epsilon: {epsilon:.4f} | "
                 f"Steps: {step_count:6d} | Time: {elapsed_time:.1f}s")
      
       # Early stopping if performing well
       if len(scores_window) >= 100 and avg_score >= 200:
           print(f"Solved in {episode} episodes! Average score: {avg_score:.2f}")
           break
  
   env.close()
   print(f"Training completed in {time.time() - start_time:.1f} seconds")
   os.makedirs('models', exist_ok=True)


   online_net.save('models/dqn_space_invaders4.keras')


   online_net.save_weights('models/dqn_weights_final.weights.h5')




   test_model(online_net, ENV_NAME)
   return online_net


if __name__ == "__main__":
   model = train()
