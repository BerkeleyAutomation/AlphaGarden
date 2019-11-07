from gym import spaces
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan
import gym
import simalphagarden
from SimAlphaGardenWrapper import SimAlphaGardenWrapper
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import time
import json
import pathlib

#TODO extract plants to wrapper
import numpy as np
from plant import Plant

NUM_TIMESTEPS = 40
NUM_X_STEPS = 3
NUM_Y_STEPS = 3
STEP = 1
SPREAD = 1
DAILY_LIGHT = 1
PLANTS_PER_COLOR = 1
# PLANT_TYPES = [((.49, .99, 0), (0.1, 30)), ((.13, .55, .13), (0.11, 30)), ((0, .39, 0), (0.13, 18))]
PLANT_TYPES = [((.49, .99, 0), (0.1, 30))]

# Creates different color plants in random locations
def get_random_plants():
    np.random.seed(285631)
    plants = []
    for color, (c1, growth_time) in PLANT_TYPES:
        x_locations = np.random.randint(1, NUM_X_STEPS - 1, (PLANTS_PER_COLOR, 1))
        y_locations = np.random.randint(1, NUM_Y_STEPS - 1, (PLANTS_PER_COLOR, 1))
        locations = np.hstack((x_locations, y_locations))
        plants.extend([Plant(row, col, c1=c1, growth_time=growth_time, color=color) for row, col in locations])
    print(plants)
    return plants

env = gym.make(
            'simalphagarden-v0', 
            wrapper_env=SimAlphaGardenWrapper(NUM_TIMESTEPS, get_random_plants(), NUM_X_STEPS, NUM_Y_STEPS, STEP, SPREAD, DAILY_LIGHT, ['basil']),
            config_file='gym-config/config.ini')
env = DummyVecEnv([lambda: env])
env = VecCheckNan(env, raise_exception=False)

# Instantiate the agent
model = PPO2(MlpPolicy, env, learning_rate=1e-8)

# Train the agent
model.learn(total_timesteps=20000)  # this will crash explaining that the invalid value originated from the env
model.save("ppo2_v2")