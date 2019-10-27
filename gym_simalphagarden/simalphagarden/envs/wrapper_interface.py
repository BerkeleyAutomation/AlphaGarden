class WrapperEnv(object):
    def __init__(self, max_time_steps):
        self.max_time_steps = max_time_steps

    '''
    Method called by the gym environment to execute an action.

    Parameters:
        action - irrigation amount to apply.  Type: np.int64
    Returns:
        state - state of the environment after irrigation
    '''
    def _take_action(self, action):
        return None

    '''
    Method called by the gym environment to reset the simulator.
    '''
    def reset(self):
        pass