class WrapperEnv(object):
    '''
    Implementing classes must specify a max_time_steps amount which is
    the number of time steps a simulator runs before resetting.
    '''
    def __init__(self, max_time_steps):
        self.max_time_steps = max_time_steps

    '''
    Method called by the gym environment to execute an action.

    Parameters:
        action - irrigation amount to apply.  Type: np.int64
    Returns:
        state - state of the environment after irrigation
    '''
    def take_action(self, action):
        pass

    '''
    Method called by the gym environment to reset the simulator.
    '''
    def reset(self):
        pass