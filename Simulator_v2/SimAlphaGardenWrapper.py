from wrapperenv import WrapperEnv
from garden import Garden
import numpy as np

class SimAlphaGardenWrapper(WrapperEnv):
    def __init__(self, max_time_steps, plants, N, M, step, spread, light_amt, plant_types):
        super(SimAlphaGardenWrapper, self).__init__(max_time_steps)
        self.plants = plants
        self.num_plants = len(plants)
        self.N = N
        self.M = M
        self.step = step
        self.spread = spread
        self.plant_types = plant_types
        self.reset()
        self.light_amt = light_amt
        self.state = self.garden.get_state()

    def get_state(self):
        return self.garden.get_state()

    def reward(self, state):
        '''TODO: Replace with actual reward function.'''
        cc_tensor = self.get_state()[self.N*self.M:]
        return np.sum(cc_tensor)

    '''
    Method called by the gym environment to execute an action.

    Parameters:
        action - list of (location, irrigation_amount) tuples, location is an
        (x, y) float64 tuple, irrigation_amount is a float64
    Returns:
        state - state of the environment after irrigation
    '''
    def take_action(self, action):
        self.garden.perform_timestep(self.light_amt, uniform_irrigation=False, irrigations=[action])
        return self.garden.get_state()

    '''
    Method called by the gym environment to reset the simulator.
    '''
    def reset(self):
        self.garden = Garden(plants=self.plants, N=self.N, M=self.M, step=self.step, spread=self.spread, plant_types=self.plant_types)