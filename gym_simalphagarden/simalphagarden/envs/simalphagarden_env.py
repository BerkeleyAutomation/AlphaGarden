import gym
from gym import error, spaces, utils
from gym.utils import seeding

class SimAlphagarden(object):
    """
    An environment wrapper for SimAlphaGarden simulators.

    The environment keeps track of the current state of the agent, updates it as
    the agent takes actions, and provides rewards to the agent.
    """

    def __init__(self):
        return

class SimAlphaGardenEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        return
    
    def step(self, action):
        return
    
    def reset(self):
        return

    def render(self, mode='human', close=False):
        return