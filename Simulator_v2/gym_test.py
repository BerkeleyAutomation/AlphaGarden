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
    return plants

env = gym.make(
            'simalphagarden-v0', 
            wrapper_env=SimAlphaGardenWrapper(NUM_TIMESTEPS, get_random_plants(), NUM_X_STEPS, NUM_Y_STEPS, STEP, SPREAD, DAILY_LIGHT, ['basil']),
            config_file='gym-config/config.ini')
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
#TODO:
# learning rate, more batches
# tensorboard
# access how reward changed over time from the learned model
# crank up learning rate until reward starts looooking fuzzy, access gpus
# higher batch size = higher learning rate it can sustain
    # try different optimal learning rate and batch size pairs
# put hyperparameters in config file
# model.learn(total_timesteps=20000)
# model.save("ppo2_simalphagarden")

model = PPO2.load("ppo2_v2")
obs = env.reset()
done = False
for i in range(50):
  e = {'obs': [], 'rewards': [], 'action': []}
  while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    e['obs'].append(obs[0].tolist())
    e['rewards'].append(rewards.item())
    e['action'].append(action[0].tolist())
    env.render()
  done = False

  pathlib.Path('PPO_Returns').mkdir(parents=True, exist_ok=True) 
  filename = 'PPO_Returns/predict_' + str(i) + '.json'
  f = open(filename, 'w')
  f.write(json.dumps(e))
  f.close()