import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class SimAlphaGardenEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, wrapper_env):
        super(SimAlphaGardenEnv, self).__init__()
        self.wrapper_env = wrapper_env
        self.max_time_steps = self.wrapper_env.max_time_steps
        # Reward ranges from 0 to 1 representing canopy cover percentage.
        self.reward_range = (0.0, 1.0)
        # Action of the format Irrigation x
        self.action_space = spaces.Discrete(100)
        # Observations include canopy cover, stomata water stress level
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 2), dtype=np.float16)
        self.reset()

    def _next_observation(self):
        return np.array([self.canopy_cover, self.water_stress])

    def _take_action(self, action):
        canopy_cover, water_stress = self.wrapper_env._take_action(action)
        self.canopy_cover = canopy_cover
        self.water_stress = water_stress

    def step(self, action):
        #Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1
        reward = self.canopy_cover
        done = self.current_step == self.max_time_steps
        obs = self._next_observation()
        return obs, reward, done, {}
    
    def reset(self):
        self.canopy_cover = 0
        self.water_stress = 0
        self.current_step = 0
        self.wrapper_env.reset()
        return self._next_observation()

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        print(f'Canopy Cover: {self.canopy_cover}')
        print(f'Water Stress: {self.water_stress}')