import numpy as np
import gym
from gym import spaces

class SimAlphaGardenEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, wrapper_env, garden_x, garden_y, garden_z, sector_rows, sector_cols,
                 action_low, action_high, obs_low, obs_high, num_plant_types, eval=False, multi=False):
        super(SimAlphaGardenEnv, self).__init__()
        self.wrapper_env = wrapper_env
        self.max_time_steps = self.wrapper_env.max_time_steps
        
        # Reward ranges from 0 to garden area
        self.reward_range = (0, sector_rows * sector_cols)

        # No action, irrigate, prune center
        self.action_space = spaces.Discrete(4)
        
        # Nubmer of plant types in the garden.  Used for reshaping global_cc_vec observation
        self.num_plant_types = num_plant_types
        
        # Observations include the seed mask for each plant type and the garden water grid
        self.observation_space = spaces.Tuple((
            spaces.Box(low=obs_low, high=obs_high, shape=(num_plant_types + 1, 1), dtype=np.float16),
            spaces.Box(low=obs_low, high=obs_high, shape=(sector_rows, sector_cols, garden_z),
                       dtype=np.float16)))

        self.eval = eval
        self.multi = multi
        self.curr_img = None
        
        self.reset()

    ''' BEGIN MULTIPROCESSING METHODS '''
    def get_centers(self):
        return self.wrapper_env.get_random_centers()
    
    def get_center_state(self, center, need_img):
        cc_img, global_cc_vec, obs = self.wrapper_env.get_center_state(center, need_img)
        if need_img: 
            return (cc_img, global_cc_vec, obs)
        return (global_cc_vec, obs)
    
    def take_multiple_actions(self, sectors, actions):
        self.wrapper_env.take_multiple_actions(sectors, actions)
    ''' END MULTIPROCESSING METHODS '''

    def _next_observation(self):
        self.sector, self.global_cc_vec, obs = self.wrapper_env.get_state(multi=self.multi)
        self.global_cc_vec = self.global_cc_vec.reshape((self.num_plant_types + 1, 1))
        return [self.global_cc_vec, obs]

    def _take_action(self, sector, action, eval=False):
        return self.wrapper_env.take_action(sector, action, self.current_step, eval)

    def get_current_step(self):
        return self.current_step

    def get_curr_action(self):
        return self.wrapper_env.get_curr_action()

    def get_sector(self):
        return self.sector

    def get_global_cc_vec(self):
        return self.global_cc_vec
    
    def step(self, action):
        state = self._take_action(self.sector, action, self.eval)
        if self.eval:
            self.curr_img, state = state
        self.reward = self.wrapper_env.reward(state)
        done = self.current_step == self.max_time_steps
        self.current_step += 1
        obs = self._next_observation()
        # print(self.current_step, reward, action, obs)
        return obs, self.reward, done, {}
    
    def reset(self):
        self.current_step = 0
        self.sector = -1
        self.curr_img = None
        self.wrapper_env.reset()
        return self._next_observation()

    def get_curr_img(self):
        return self.curr_img

    def get_garden_state(self):
        return self.wrapper_env.get_garden_state()

    def get_radius_grid(self):
        return self.wrapper_env.get_radius_grid()

    def get_metrics(self):
        return self.wrapper_env.get_metrics()
    
    def get_prune_window_greatest_width(self):
        return self.wrapper_env.get_prune_window_greatest_width(self.sector)

    def show_animation(self):
        return self.wrapper_env.show_animation()

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        print(f'Reward: {self.reward}')
