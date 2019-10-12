from gym.envs.registration import register

register(
    id='simalphagarden-v0',
    entry_point='simalphagarden.envs:SimAlphaGardenEnv'
)