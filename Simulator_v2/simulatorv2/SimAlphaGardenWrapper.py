from wrapperenv import WrapperEnv
from simulatorv2.garden import Garden
from simulatorv2.plant_type import PlantType
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
        self.curr_action = np.zeros((N*M,))
         
        self.config = configparser.ConfigParser()
        self.config.read('gym_config/config.ini')

    def get_state(self):
        return self.garden.get_state()

    def get_garden_state(self):
        return self.garden.get_garden_state()

    def get_radius_grid(self):
        return self.garden.get_radius_grid()

    def get_curr_action(self):
        return self.curr_action

    def reward(self, state):
        total_cc = np.sum(self.garden.leaf_grid)
        cc_per_plant = [np.sum(self.garden.leaf_grid[:,:,i]) for i in range(self.garden.leaf_grid.shape[2])]
        prob = cc_per_plant / total_cc
        prob = prob[np.where(prob > 0)]
        entropy = -np.sum(prob*np.log(prob))
        water_coef = self.config.getfloat('cnn', 'water_coef')
        cc_coef = self.config.getfloat('cnn', 'cc_coef')
        action_sum = self.N * self.M 
        return (cc_coef * total_cc) + (0 * entropy) + water_coef * np.sum(-1 * self.curr_action/action_sum + 1)
        
    '''
    Method called by the gym environment to execute an action.

    Parameters:
        action - list of (location, irrigation_amount) tuples, location is an
        (x, y) float64 tuple, irrigation_amount is a float64
    Returns:
        state - state of the environment after irrigation
    '''
    def take_action(self, action):
        self.curr_action = action
        #print('ACTION', action)
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
                irr_threshold=0,
                step=self.step,
                plant_types=self.PlantType.get_n_names(self.num_plant_types),
                animate=False)

    '''
    Method called by the environment to display animations.
    '''
    def show_animation(self):
        self.garden.show_animation()
