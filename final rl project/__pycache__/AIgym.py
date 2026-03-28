import gymnasium as gym
import numpy as np
import ale_py
import cv2
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformObservation
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Hyperparameters
ENV_NAME       = "ALE/SpaceInvaders-v5"
MEMORY_SIZE    = 100000
BATCH_SIZE     = 32
GAMMA          = 0.99
LEARNING_RATE  = 1e-4
EPSILON_START  = 1.0
EPSILON_MIN    = 0.1
EPSILON_DECAY  = 1e-6
TARGET_UPDATE  = 10  # episodes
EPISODES       = 500
STACK_SIZE     = 4
IMG_HEIGHT     = 84
IMG_WIDTH      = 84

# Preprocessing: grayscale, resize, normalize


# Build Q-network
def build_model(input_shape, n_actions):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    model.add(Conv2D(8, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

# Main training loop
def train():
    # Base environment
    env = gym.make(ENV_NAME, frameskip=1, rendermode="")
    # Wrap for preprocessing: grayscale, resize to 84×84, frame skip
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip = 4)
    #env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
    # Wrap for stacking last 4 frames
    env = FrameStackObservation(env, stack_size=4)
    env = TransformObservation(env, lambda obs: np.transpose(obs, (1, 2, 0)), env.observation_space)



    n_actions = env.action_space.n
    input_shape = (IMG_HEIGHT, IMG_WIDTH, STACK_SIZE)

    # Initialize networks
    online_net = build_model(input_shape, n_actions)
    target_net = build_model(input_shape, n_actions)
    target_net.set_weights(online_net.get_weights())

    # Replay memory
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START

    for episode in range(1, EPISODES+1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                q_vals = online_net.predict(state[np.newaxis, :], verbose=0)
                action = np.argmax(q_vals[0])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Store transition
            memory.append((state, action, reward, next_state, done))
            state = next_state

            # Learning step
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb = map(np.array, zip(*batch))

                # Predict Q-values
                q_next = target_net.predict(next_states_mb, verbose=0)
                q_target = online_net.predict(states_mb, verbose=0)

                for i in range(BATCH_SIZE):
                    if dones_mb[i]:
                        q_target[i, actions_mb[i]] = rewards_mb[i]
                    else:
                        q_target[i, actions_mb[i]] = rewards_mb[i] + GAMMA * np.max(q_next[i])

                online_net.train_on_batch(states_mb, q_target)

            # Decay epsilon
            if epsilon > EPSILON_MIN:
                epsilon -= EPSILON_DECAY

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.set_weights(online_net.get_weights())
        
        print(f"Episode: {episode}, Score: {total_reward}, Epsilon: {epsilon:.4f}")
        
    env.close()
    return

if __name__ == "__main__":
    train()
