from gym.envs.registration import register
import yaml

register(
    id='fastag-v0',
    entry_point='gym_fastag.envs:FastAg',
    kwargs={}
)
