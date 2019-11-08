import numpy as np
import gym
from gym import spaces

class SimAlphaGardenEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, wrapper_env, garden_x, garden_y, garden_z, action_low, action_high, obs_low, obs_high):
        super(SimAlphaGardenEnv, self).__init__()
        self.wrapper_env = wrapper_env
        self.max_time_steps = self.wrapper_env.max_time_steps

        # Reward ranges from 0 to garden area.
        self.reward_range = (0, garden_x * garden_y)

        # There is one action for every cell in the garden.
        num_actions = garden_x * garden_y
        self.action_space = spaces.Box(low=np.array([action_low for i in range(num_actions)]), high=np.array([action_high for i in range(num_actions)]), dtype=np.float16)
        
        # Observations include canopy cover for each plant in the garden
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(garden_x, garden_y, garden_z), dtype=np.float16)
        
        self.reset()

    def _next_observation(self):
        return self.wrapper_env.get_state()

    def _take_action(self, action):
        return self.wrapper_env.take_action(action)

    def step(self, action):
        state = self._take_action(action)

        self.current_step += 1

        self.reward = self.wrapper_env.reward(state)
        done = self.current_step == self.max_time_steps
        obs = self._next_observation()
        # print(self.current_step, reward, action, obs)
        return obs, self.reward, done, {}
    
    def reset(self):
        self.current_step = 0
        self.wrapper_env.reset()
        return self._next_observation()

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        print(f'Reward: {self.reward}')