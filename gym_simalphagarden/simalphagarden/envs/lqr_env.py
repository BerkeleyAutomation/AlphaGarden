import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class LqrEnv(gym.Env):

    def __init__(self, size, init_state, state_bound):
        self.init_state = init_state
        self.size = size 
        self.action_space = spaces.Box(low=-state_bound, high=state_bound, shape=(size,))
        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=(size,))
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,u):
        costs = np.sum(u**2) + np.sum(self.state**2)
        self.state = np.clip(self.state + u, self.observation_space.low, self.observation_space.high)
        return self._get_obs(), -costs, False, {}

    def _reset(self):
        high = self.init_state*np.ones((self.size,))
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state