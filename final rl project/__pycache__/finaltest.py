import gymnasium as gym
import ale_py
import numpy as np

# Monkey-patch model_from_config once for keras-rl2
import tensorflow.keras.models as _kmodels
_kmodels.model_from_config = _kmodels.model_from_json

# RL imports
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

# Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


def make_atari_env(env_name):
    """Create and wrap Atari environment with standard preprocessing"""
    env = gym.make(env_name, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,          # skip 4 frames per action
        grayscale_obs=True,    # convert to grayscale
        scale_obs=True         # scale pixels to [0,1]
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)  # now obs.shape = (84,84,4)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.moveaxis(obs, 0, -1), env.observation_space)
    
    return env


def build_model(h, w, c, n_actions):
    """Build CNN model for DQN agent"""
    model = Sequential()
    model.add(InputLayer(shape=(h, w, c)))
    # Fixed: Added missing model.add() for first Conv2D layer
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu', name='conv1'))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu', name='conv2'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu', name='conv3'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='fc1'))
    model.add(Dense(n_actions, activation='linear', name='output'))
    
    return model


def build_agent(model, n_actions):
    """Build DQN agent with specified hyperparameters"""
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(), attr='eps',
        value_max=1.0, value_min=0.1, value_test=0.05,
        nb_steps=1_000_000
    )
    # window_length=1 because FrameStack already gives us 4 frames per obs
    memory = SequentialMemory(limit=1_000_000, window_length=1)
    dqn = DQNAgent(
        model=model,
        nb_actions=n_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=50_000,
        target_model_update=10_000,
        gamma=0.99,
        enable_dueling_network=True,
        dueling_type='avg',
    )
    return dqn


# Create environment and get dimensions
env = make_atari_env("ALE/SpaceInvaders-v5")
height, width, channels = env.observation_space.shape  # => (84,84,4)
actions = env.action_space.n

print(f"Environment: Space Invaders")
print(f"Observation shape: {env.observation_space.shape}")
print(f"Number of actions: {actions}")

# Build and compile model
model = build_model(height, width, channels, actions)
model.compile(optimizer=Adam(learning_rate=2.5e-4), metrics=['mae'])

# Print model summary
print("\nModel Architecture:")
model.summary()

# Build and compile agent
dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=2.5e-4), metrics=['mae'])

print("\nStarting training...")
print("This will take several hours to complete!")

# Train for 5M steps with periodic checkpoints
try:
    history = dqn.fit(
        env, 
        nb_steps=5_000_000, 
        visualize=False, 
        verbose=2,
        # Save weights every 500k steps
        callbacks=[
            ModelCheckpoint(
                'dqn_spaceinvaders_checkpoint_{step}.h5f',
                save_freq=500_000,
                save_weights_only=True
            )
        ]
    )
    
    print("Training completed successfully!")
    
    # Save final weights
    dqn.save_weights('dqn_spaceinvaders_final_weights.h5f', overwrite=True)
    print("Final weights saved!")
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    dqn.save_weights('dqn_spaceinvaders_interrupted_weights.h5f', overwrite=True)
    print("Weights saved before interruption!")

# Evaluate the trained agent
print("\nEvaluating agent...")
scores = dqn.test(env, nb_episodes=10, visualize=True)
print(f"Average score over 10 episodes: {np.mean(scores.history['episode_reward']):.2f}")
print(f"Best score: {np.max(scores.history['episode_reward']):.2f}")
print(f"Worst score: {np.min(scores.history['episode_reward']):.2f}")

# Optional: Save training history if available
try:
    import pickle
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("Training history saved!")
except:
    print("Could not save training history")

env.close()