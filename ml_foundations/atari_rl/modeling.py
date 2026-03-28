from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_dqn_cnn(
    input_shape: tuple[int, int, int],
    n_actions: int,
    *,
    learning_rate: float = 2.5e-4,
    dense_units: int = 512,
):
    """Build a stable DQN-style CNN used across multiple legacy experiments."""
    model = Sequential(
        [
            InputLayer(shape=input_shape),
            Conv2D(32, (8, 8), strides=(4, 4), activation="relu", padding="valid"),
            Conv2D(64, (4, 4), strides=(2, 2), activation="relu", padding="valid"),
            Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="valid"),
            Flatten(),
            Dense(dense_units, activation="relu"),
            Dense(n_actions, activation="linear", dtype="float32"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="huber",
    )
    return model


def clone_target_network(online_network: tf.keras.Model):
    """Clone a target network and copy weights from the online network."""
    target = tf.keras.models.clone_model(online_network)
    target.build(online_network.input_shape)
    target.set_weights(online_network.get_weights())
    target.compile(
        optimizer=online_network.optimizer.__class__.from_config(
            online_network.optimizer.get_config()
        ),
        loss=online_network.loss,
    )
    return target
