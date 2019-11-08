from wrapperenv import WrapperEnv
from garden import Garden
from plant_type import PlantType
import numpy as np

class SimAlphaGardenWrapper(WrapperEnv):
    def __init__(self, max_time_steps, N, M, num_plant_types, num_plants_per_type, step=1):
        super(SimAlphaGardenWrapper, self).__init__(max_time_steps)
        self.N = N
        self.M = M
        self.step = step
        self.num_plant_types = num_plant_types
        self.num_plants_per_type = num_plants_per_type
        self.PlantType = PlantType()
        self.reset()
        self.state = self.garden.get_state()

    def get_state(self):
        return self.garden.get_state()

    def reward(self, state):
        total_cc = 0
        for row_matrix in state:
            # Iterate over columns
            for i, column in enumerate(row_matrix.T):
                if i != len(row_matrix.T) - 1:
                    total_cc += sum(column)
        return total_cc

    '''
    Method called by the gym environment to execute an action.

    Parameters:
        action - list of (location, irrigation_amount) tuples, location is an
        (x, y) float64 tuple, irrigation_amount is a float64
    Returns:
        state - state of the environment after irrigation
    '''
    def take_action(self, action):
        self.garden.perform_timestep(uniform_irrigation=False, irrigations=[action])
        return self.garden.get_state()

    '''
    Method called by the gym environment to reset the simulator.
    '''
    def reset(self):
        self.garden = \
            Garden(
                plants=self.PlantType.get_random_plants(self.PlantType.get_n_types(self.num_plant_types), self.M, self.N, self.num_plants_per_type),
                N=self.N,
                M=self.M,
                step=self.step,
                plant_types=self.PlantType.get_n_names(self.num_plant_types))
