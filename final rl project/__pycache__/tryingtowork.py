import gymnasium as gym
import ale_py
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
# Monkey-patch courtesey of stack over-flow
import tensorflow.keras.models as _kmodels
import json

def _patched_model_from_json(json_string, custom_objects=None):
    if isinstance(json_string, dict):
        # If it's already a dict, use it directly
        config = json_string
    else:
        # If it's a string, parse it
        config = json.loads(json_string)
    return _patched_model_from_config(config, custom_objects=custom_objects)

def _patched_model_from_config(config, custom_objects=None):
    """Create model from config dict - using a simpler approach"""
    try:
        # Try the modern Keras way first
        from tensorflow.keras.models import model_from_json
        return model_from_json(json.dumps(config), custom_objects=custom_objects)
    except:
        # Fallback: try to reconstruct manually (this is a simplified version)
        from tensorflow.keras.models import Model, Sequential
        from tensorflow.keras.layers import deserialize as layer_deserialize
        
        if config.get('class_name') == 'Sequential':
            model = Sequential()
            for layer_config in config['config']['layers']:
                layer = layer_deserialize(layer_config, custom_objects=custom_objects)
                model.add(layer)
            return model
        elif config.get('class_name') == 'Functional':
            # For functional models, this is more complex - let's use a different approach
            raise NotImplementedError("Functional model cloning not fully implemented in fallback")
        else:
            raise ValueError(f"Unknown model class: {config.get('class_name')}")

# Apply patches
_kmodels.model_from_json = _patched_model_from_json
_kmodels.model_from_config = _patched_model_from_config

# Also patch the clone_model function directly to avoid the issue
import rl.util as _rl_util

def _patched_clone_model(model, custom_objects=None):
    """Simplified clone_model that works with newer Keras"""
    from tensorflow.keras.models import clone_model
    try:
        # Use Keras built-in clone_model
        return clone_model(model)
    except:
        # Fallback: create a new model with same architecture
        config = model.get_config()
        new_model = _patched_model_from_config(config, custom_objects)
        return new_model

_rl_util.clone_model = _patched_clone_model
# RL imports
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

# Keras imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam

import tensorflow.keras.callbacks     as _pub_cb
from tensorflow.python.keras.callbacks import Callback as _priv_cb

# Add all three hook‐check methods so CallbackList won’t crash on History, BaseLogger, etc.
for fn in (
    "_implements_train_batch_hooks",
    "_implements_test_batch_hooks",
    "_implements_predict_batch_hooks",
):
    setattr(_pub_cb.Callback, fn, lambda self: False)
    setattr(_priv_cb,          fn, lambda self: False)

from tensorflow.keras.models import Model as _KModel
_KModel.reset_states = lambda self: None

def make_atari_env(env_name):
    env = gym.make(env_name, frameskip =1)
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,          # skip 4 frames per action
        grayscale_obs=True,    # convert to grayscale
        scale_obs=True         # scale pixels to [0,1]
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)  # now obs.shape = (84,84,4)
    return env



def build_model(h, w, c, n_actions):
    # Use Functional API instead of Sequential - this should work better with keras-rl2
    inputs = Input(shape=(c, h, w))
    
    x = Conv2D(32, (8, 8), strides=4, activation='relu',data_format='channels_first')(inputs)
    x = Conv2D(64, (4, 4), strides=2, activation='relu',data_format='channels_first')(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu',data_format='channels_first')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(n_actions, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


env = make_atari_env("ALE/SpaceInvaders-v5")
print("Obs space shape:", env.observation_space.shape)

channels, height, width = env.observation_space.shape
actions = env.action_space.n
model = build_model(height, width, channels, actions)
def build_agent(model, n_actions):
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
        enable_dueling_network=False, 
    )
    return dqn


# Now build the agent - this should work without errors

dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=2.5e-4), metrics=['mae'])

# Train for 5M steps (adjust as you like)
dqn.fit(env, nb_steps=5_000_000, visualize=False, verbose=2)

# Evaluate
scores = dqn.test(env, nb_episodes=10, visualize=True)
print("Average score:", np.mean(scores.history['episode_reward']))

# Save weights
dqn.save_weights('dqn_spaceinvaders_weights.h5f', overwrite=True)
