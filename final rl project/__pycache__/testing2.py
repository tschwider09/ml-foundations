import gymnasium as gym
import ale_py
import numpy as np

# Monkey-patch for keras-rl2 compatibility with newer Keras versions
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