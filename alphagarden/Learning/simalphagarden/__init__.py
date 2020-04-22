from gym.envs.registration import register
from gym.envs import registry
import numpy as np

id = 'simalphagarden-v0'
all_envs = [env_spec.id for env_spec in registry.all()]
if id not in all_envs:
   register(
       id='simalphagarden-v0',
       entry_point='simalphagarden.envs:SimAlphaGardenEnv'
   )