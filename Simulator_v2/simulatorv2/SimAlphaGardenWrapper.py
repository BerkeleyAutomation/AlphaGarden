from wrapperenv import WrapperEnv
from simulatorv2.garden import Garden
from simulatorv2.plant_type import PlantType
import numpy as np
import configparser
import matplotlib.pyplot as plt


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    buf = np.array(fig.canvas.renderer.buffer_rgba())
    return buf

class SimAlphaGardenWrapper(WrapperEnv):
    def __init__(self, max_time_steps, N, M, sector_width, sector_height, num_plant_types, num_plants_per_type, step=1):
        super(SimAlphaGardenWrapper, self).__init__(max_time_steps)
        self.N = N
        self.M = M
        self.num_sectors = (M * N) / (sector_width * sector_height)
        self.sector_width = sector_width
        self.sector_height = sector_height
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
    
    def get_sector_x(self, sector):
        return (sector % (self.N // self.sector_width)) * self.sector_width
    
    def get_sector_y(self, sector):
        return (sector // (self.M // self.sector_height)) * self.sector_height

    ''' Returns sector number and state associated with the sector. '''
    def get_random_sector(self):
        # TODO: Need to seed numpy?
        sector = np.random.randint(low=0, high=self.num_sectors, size=1)[0]
        full_state = self.garden.get_state()
        x = self.get_sector_x(sector)
        y = self.get_sector_y(sector)
        return sector, full_state[x:x+self.sector_width,y:y+self.sector_height,:]

    def get_canopy_image(self, sector):
        x_low, y_low = self.get_sector_x(sector), self.get_sector_y(sector)
        x_high, y_high = x_low + self.sector_width, y_low + self.sector_height
        fig, ax = plt.subplots()
        plt.xlim((0, self.sector_width * self.garden.step))
        plt.ylim((0, self.sector_height * self.garden.step))
        ax.set_aspect('equal')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        shapes = []
        for plant in sorted([plant for plant_type in self.garden.plants for plant in plant_type.values()],
                            key=lambda x: x.height, reverse=True):
            if x_low <= plant.row <= x_high and y_low <= plant.height <= y_high:
                if plant.pruned:
                    shape = plt.Rectangle((plant.row * self.garden.step - plant.radius,
                                           plant.col * self.garden.step - plant.radius), plant.radius * 2, plant.radius * 2,
                                          fc='red', ec='red')
                else:
                    shape = plt.Circle((plant.row, plant.col) * self.garden.step, plant.radius, color=plant.color)
                shape_plot = ax.add_artist(shape)
                shapes.append(shape_plot)
        plt.tight_layout()
        plt.show()
        # img_as_nparray = fig2data(fig)
        # return img_as_nparray

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
    def take_action(self, sector, action):
        self.curr_action = action
        #print('ACTION', action)
        self.garden.perform_timestep(sector=sector, irrigations=action)
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
