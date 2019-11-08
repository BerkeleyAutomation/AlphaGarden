#!/usr/bin/env python3

import gym
import simalphagarden
from SimAlphaGardenWrapper import SimAlphaGardenWrapper
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines import PPO2
import time
import json
import pathlib
import configparser
from shutil import copyfile
from baseline_policy import baseline_policy

#TODO extract plants to wrapper
import numpy as np
from plant import Plant


config = configparser.ConfigParser()
config.read('gym-config/config.ini')

NUM_TIMESTEPS = 40
NUM_X_STEPS = config.getint('garden', 'X')
NUM_Y_STEPS = config.getint('garden', 'Y')
STEP = 1
SPREAD = 1
DAILY_LIGHT = 1
PLANTS_PER_COLOR = config.getint('garden', 'num_plants_per_type')
PLANT_TYPES = [((.49, .99, 0), (0.1, 30)), ((.13, .55, .13), (0.11, 30)), ((0, .39, 0), (0.13, 18))][:config.getint('garden', 'num_plant_types')]

# Creates different color plants in random locations
# TODO: make random
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
            wrapper_env=SimAlphaGardenWrapper(NUM_TIMESTEPS, get_random_plants(), NUM_X_STEPS, NUM_Y_STEPS, STEP, SPREAD, DAILY_LIGHT, ['basil' for i in range(len(PLANT_TYPES))]),
            config_file='gym-config/config.ini')
env = DummyVecEnv([lambda: env])
env = VecCheckNan(env, raise_exception=False)

# Instantiate the agent
model = PPO2(MlpPolicy, env, learning_rate=1e-8, verbose=1, tensorboard_log="./ppo_v2_tensorboard/")

# Train the agent
# model.learn(total_timesteps=200000)  # this will crash explaining that the invalid value originated from the env
# model_name = 'ppo2_v2_' + time.strftime('%Y-%m-%d-%H-%M-%S')
# model.save(model_name)
# pathlib.Path('PPO_Configs').mkdir(parents=True, exist_ok=True)
# copyfile('./gym-config/config.ini', './PPO_Configs/' + model_name)

baseline = True
model = PPO2.load("ppo2_v2_2019-11-07-17-53-22")
obs = env.reset()
done = False
for i in range(1):
    e = {'obs': [], 'rewards': [], 'action': []}
    while not done:
        if baseline:
            action = baseline_policy(obs, STEP, 0.5, 0.5, 5)
        else:
            action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        e['obs'].append(obs[0].tolist())
        e['rewards'].append(rewards.item())
        e['action'].append(action[0].tolist())
        env.render()
    done = False

    if baseline:
        path = 'Baseline_Returns'
    else:
        path = 'PPO_Returns'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    filename = path + '/predict_' + str(i) + '.json'
    f = open(filename, 'w')
    f.write(json.dumps(e))
    f.close()
