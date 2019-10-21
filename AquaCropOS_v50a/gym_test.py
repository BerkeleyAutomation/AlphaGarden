import gym
import simalphagarden
from aquacropos_wrapper import AquaCropOSWrapper
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import time
import json
import pathlib

env = gym.make('simalphagarden-v0', wrapper_env=AquaCropOSWrapper())
env = DummyVecEnv([lambda: env])

# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=20000)
# model.save("ppo2_simalphagarden")

model = PPO2.load("PPO_Models/ppo2_simalphagarden_100_mm")
obs = env.reset()
done = False
for i in range(50):
  e = {'obs_cc': [], 'obs_ws': [], 'rewards': [], 'action': []}
  while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    e['obs_cc'].append(obs[0][0][0].item())
    e['obs_ws'].append(obs[0][0][1].item())
    e['rewards'].append(rewards.item())
    e['action'].append(action.item())
    env.render()
  done = False
  pathlib.Path('PPO_Returns').mkdir(parents=True, exist_ok=True) 
  filename = 'PPO_Returns/predict_' + str(i) + '.json'
  f = open(filename, 'w')
  f.write(json.dumps(e))
  f.close()