from gym.envs.registration import register
import numpy as np

register(
    id='simalphagarden-v0',
    entry_point='simalphagarden.envs:SimAlphaGardenEnv'
)

# register(
#     id='Lqr-v0',
#     entry_point='simalphagarden.envs:LqrEnv',
#     max_episode_steps=150,
#     kwargs={'size': 1, 'init_state': 10., 'state_buond': np.inf}
# )