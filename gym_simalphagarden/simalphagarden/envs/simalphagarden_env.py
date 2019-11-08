import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import configparser

class SimAlphaGardenEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, wrapper_env, config_file):
        super(SimAlphaGardenEnv, self).__init__()
        self.wrapper_env = wrapper_env
        self.max_time_steps = self.wrapper_env.max_time_steps
        config = configparser.ConfigParser()
        config.read(config_file)
        # Reward ranges from 0 to 1 representing canopy cover percentage.
        self.reward_range = (0, config.getfloat('garden', 'X') * config.getfloat('garden', 'Y'))
        # Action of the format Irrigation x
        # self.action_space = spaces.Discrete(config.getint('action', 'range'))
        action_range = config.getint('garden', 'X') * config.getint('garden', 'Y')
        self.action_space = spaces.Box(low=np.array([0.0 for i in range(action_range)]), high=np.array([config.getfloat('action', 'high') for i in range(action_range)]), dtype=np.float16)
        # Observations include canopy cover for each plant in the garden
        self.observation_space = spaces.Box(low=config.getint('obs', 'low'), high=config.getint('obs', 'high'), shape=(config.getint('garden', 'X'), config.getint('garden', 'Y'), config.getint('garden', 'num_plant_types') + 1), dtype=np.float16)
        self.reset()

    def _next_observation(self):
        return self.wrapper_env.get_state()

    def _take_action(self, action):
        return self.wrapper_env.take_action(action)

    def step(self, action):
        state = self._take_action(action)
        self.current_step += 1
        reward = self.wrapper_env.reward(state)
        done = self.current_step == self.max_time_steps
        obs = self._next_observation()
        print(self.current_step, reward, action, obs)
        return obs, reward, done, {}
    
    def reset(self):
        self.current_step = 0
        self.wrapper_env.reset()
        return self._next_observation()

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')