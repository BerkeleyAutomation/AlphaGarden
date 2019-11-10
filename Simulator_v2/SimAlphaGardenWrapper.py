from wrapperenv import WrapperEnv
from garden import Garden
from plant_type import PlantType
import numpy as np
import configparser

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
        self.cumulative_action = np.zeros((N*M,))
         
        self.config = configparser.ConfigParser()
        self.config.read('gym_config/config.ini')

    def get_state(self):
        return self.garden.get_state()

    def get_garden_state(self):
        return self.garden.get_garden_state()

    def reward(self, state):
        total_cc = np.sum(self.garden.leaf_grid)
        cc_per_plant = [np.sum(self.garden.leaf_grid[:,:,i]) for i in range(self.garden.leaf_grid.shape[2])]
        prob = cc_per_plant / total_cc
        entropy = -np.sum(prob*np.log(prob))
        water_coef = self.config.getfloat('cnn', 'water_coef')
        return total_cc + (0 * entropy) - (water_coef * np.sum(self.cumulative_action)) ## TODO: ADD Lambdas!!!!
        
    '''
    Method called by the gym environment to execute an action.

    Parameters:
        action - list of (location, irrigation_amount) tuples, location is an
        (x, y) float64 tuple, irrigation_amount is a float64
    Returns:
        state - state of the environment after irrigation
    '''
    def take_action(self, action):
        self.cumulative_action += action
        self.garden.perform_timestep(irrigations=action)
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
