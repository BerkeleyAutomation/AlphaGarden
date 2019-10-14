import gym
import simalphagarden
from aquacropos_wrapper import AquaCropOSWrapper
env = gym.make('simalphagarden-v0', wrapper_env=AquaCropOSWrapper())
