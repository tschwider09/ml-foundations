from tensorflow.keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
import argparse

from .environment import create_atari_env

def build_model(height, width, channels, actions):
    """Build a lightweight CNN for Atari frame stacks."""
    model = Sequential()
    model.add(
        Convolution2D(
            16,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(height, width, channels),
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(((height * width) * 8), activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def run_random_policy(episodes=5, env_name="ALE/SpaceInvaders-v5", render_mode="human"):
    """
    Smoke-test an Atari environment by sampling random actions.
    Reference: https://gymnasium.farama.org/introduction/basic_usage/
    """
    env = create_atari_env(env_name, render_mode=render_mode, frame_skip=4, stack_size=4)
    observation, _ = env.reset()
    height, width, channels = observation.shape
    actions = env.action_space

    print(f"Action meanings: {env.unwrapped.get_action_meanings()}")
    print(f"Observation shape: {(height, width, channels)}")

    for episode in range(episodes):
        _, _ = env.reset()
        done = False
        score = 0
        frames = 0
        non_zero_rewards = 0

        while not done:
            action = actions.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            score += reward
            frames += 1
            if reward != 0:
                non_zero_rewards += 1
            done = terminated or truncated

        print(
            f"Episode: {episode + 1} "
            f"Score: {score} Frames: {frames} Non-zero rewards: {non_zero_rewards}"
        )

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Run a random-policy Space Invaders smoke test.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable human rendering (off by default for headless compatibility).",
    )
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    run_random_policy(episodes=args.episodes, render_mode=render_mode)


if __name__ == "__main__":
    main()
