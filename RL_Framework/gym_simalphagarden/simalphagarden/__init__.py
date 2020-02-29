from gym.envs.registration import register
import numpy as np

register(
    id='simalphagarden-v0',
    entry_point='simalphagarden.envs:SimAlphaGardenEnv'
)