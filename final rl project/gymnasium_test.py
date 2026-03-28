import gymnasium as gym
import ale_py
import numpy as np
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam

#from tensorforce import Agent, Environment



# I am using dompamine from google for rl
def build_model(height, width, channels, actions):

    #reduce the model size
    model = Sequential()
    model.add(Convolution2D(16, (3, 3), activation="relu", padding="same", input_shape=(height, width, channels)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(((height*width)*8), activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model

"Cited: https://gymnasium.farama.org/introduction/basic_usage/" 
env = gym.make("ALE/TimePilot-v5", render_mode="human")
#env = gym.make("ALE/TimePilot-v5")
observation, info = env.reset()
height, width, channels = env.observation_space.shape
actions = env.action_space
print(env.unwrapped.get_action_meanings())


episodes = 5
#episodes
for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0
    frames = 0
    scores = 0
    while not done:
        env.render()
        action = actions.sample()  
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        frames += 1
        if reward != 0:
            scores+=1
        done = terminated or truncated
    print(f"Episode: {episode+1} Score: {score} Frames: {frames} Scores: {scores}")

env.close()