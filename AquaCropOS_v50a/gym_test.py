import gym
import simalphagarden
from aquacropos_wrapper import AquaCropOSWrapper
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('simalphagarden-v0', wrapper_env=AquaCropOSWrapper())
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)
model.save("ppo2_simalphagarden")

obs = env.reset()
for i in range(2000):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render()