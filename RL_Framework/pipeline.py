#!/usr/bin/env python3
import gym
import simalphagarden
import time
import configparser
import numpy as np
import json
import pathlib
from cnn_policy import CustomCnnPolicy
from shutil import copyfile
from simulatorv2.SimAlphaGardenWrapper import SimAlphaGardenWrapper
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
import cProfile
import pstats
import io
from graph_utils import GraphUtils
from file_utils import FileUtils

def get_sector_x(sector, garden_x, sector_width):
    return (sector % (garden_x // sector_width)) * sector_width

def get_sector_y(sector, garden_y, sector_height):
    return (sector // (garden_y // sector_height)) * sector_height

class Pipeline:
    def __init__(self):
        self.graph_utils = GraphUtils()
        self.file_utils = FileUtils()

    def evaluate_policy(self, folder_path, num_evals, env, garden_x, garden_y, sector_width, sector_height, is_baseline=False, baseline_policy=None, step=1):
        model = None
        if not is_baseline:
            model = PPO2.load('./' + folder_path + '/model')
        done = False
        for _ in range(num_evals):
            obs = env.reset()
            garden_obs = env.env_method('get_garden_state')
            e = {'cell_avg_action': [], 'full_state_action': [], 'full_state': [], 'rewards': [], 'action': []}

            cell_avg_action = {}
            for x in range(garden_x):
                for y in range(garden_y):
                    cell_avg_action[x, y] = 0

            step_counter = 0
            idx = 0
            while not done:
                print("ITERATION ", idx)
                action = None
                if is_baseline:
                    action = baseline_policy(obs, step, threshold=0.5, amount=1, irr_threshold=0)
                else:
                    action, _states = model.predict(obs)
                obs, rewards, done, _ = env.step(action)
                idx += 1
                action = env.env_method('get_curr_action')
                garden_obs = env.env_method('get_garden_state')
                radius_grid = env.env_method('get_radius_grid')

                # if not done:
                #     step_counter = env.env_method('get_current_step')[0]

                #     rg_list = radius_grid[0].tolist()
                #     obs_action_pairs = []
                    
                #     sector = env.env_method('get_sector')[0]
                #     sector_x = get_sector_x(sector, garden_x, sector_width)
                #     sector_y = get_sector_y(sector, garden_y, sector_height)
                #     for x in range(garden_x):
                #         for y in range(garden_y):
                #             cell = (x, y)
                #             if cell[0] >= sector_x and cell[0] < sector_x + sector_width and cell[1] >= sector_y and cell[1] < sector_y + sector_height:
                #                 cell_action = env.env_method('get_irr_action')[0]
                #             else:
                #                 cell_action = 0
                #             obs_action_pairs.append({str(cell) : (str(rg_list[x][y][0]), str(cell_action))})
                #             cell_avg_action[cell] += cell_action
                #     e['full_state_action'].append({step_counter : obs_action_pairs})

                #     e['full_state'].append(garden_obs[0].tolist())
                #     e['rewards'].append(rewards.item())
                #     e['action'].append(action[0].tolist())
                #     env.render()
            done = False

            # for x in range(garden_x):
            #     for y in range(garden_y):
            #         cell_avg_action[(x, y)] /= step_counter
            #         e['cell_avg_action'].append({str((x, y)) : cell_avg_action[(x, y)], 'final_radius': rg_list[x][y][0]})
            
            ''' UNCOMMENT IF YOU WANT TO WRITE OUTPUTS TO A FILE.  WILL TAKE AWHILE FOR LARGE GARDENS. '''
            # pathlib.Path(folder_path + '/Returns').mkdir(parents=True, exist_ok=True)
            # filename = folder_path + '/Returns' + '/predict_' + str(i) + '.json'
            # f = open(filename, 'w')
            # f.write(json.dumps(e, indent=4))
            # f.close()

    def single_run(self, folder_path, num_evals, policy_kwargs=None, is_baseline=False, baseline_policy=None):
        # initialize cProfile
        profiler_object = cProfile.Profile()
        profiler_object.enable()

        config = configparser.ConfigParser()
        config.read('gym_config/config.ini')

        rl_time_steps = config.getint('rl', 'time_steps')
        ent_coef = config.getfloat('rl', 'ent_coef')
        n_steps = config.getint('rl', 'n_steps')
        nminibatches = config.getint('rl', 'nminibatches')
        noptepochs = config.getint('rl', 'noptepochs')
        learning_rate = config.getfloat('rl', 'learning_rate')
        time_steps = config.getint('garden', 'time_steps')
        step = config.getint('garden', 'step')
        num_plants_per_type = config.getint('garden', 'num_plants_per_type')
        num_plant_types = config.getint('garden', 'num_plant_types')
        garden_x = config.getint('garden', 'X')
        garden_y = config.getint('garden', 'Y')
        garden_z = 2 * config.getint('garden', 'num_plant_types') + 1 # Z axis contains a matrix for every plant type plus one for water levels.
        sector_width = config.getint('garden', 'sector_width')
        sector_height = config.getint('garden', 'sector_height')
        action_low = config.getfloat('action', 'low')
        action_high = config.getfloat('action', 'high')
        obs_low = config.getint('obs', 'low')
        obs_high = config.getint('obs', 'high')

        env = gym.make(
                    'simalphagarden-v0',
                    wrapper_env=SimAlphaGardenWrapper(time_steps, garden_x, garden_y, sector_width, sector_height, num_plant_types, num_plants_per_type, step=step),
                    garden_x=garden_x,
                    garden_y=garden_y,
                    garden_z=garden_z,
                    sector_width=sector_width,
                    sector_height=sector_height,
                    action_low=action_low,
                    action_high=action_high,
                    obs_low=obs_low,
                    obs_high=obs_high,
                    )
        env = DummyVecEnv([lambda: env])
        # TODO: Normalize input features? VecNormalize
        env = VecCheckNan(env, raise_exception=False)

        if is_baseline:
            copyfile('gym_config/config.ini', folder_path + '/config.ini')

            # Evaluate baseline on 50 random environments of same parameters.
            self.evaluate_policy(folder_path, num_evals, env, garden_x, garden_y, sector_width, sector_height, is_baseline=True, baseline_policy=baseline_policy, step=1)

            # Graph evaluations
            self.graph_utils.graph_evaluations(folder_path, garden_x, garden_y, time_steps, step, num_evals, num_plant_types)
        else:
            pathlib.Path(folder_path + '/ppo_v2_tensorboard').mkdir(parents=True, exist_ok=True)
            # Instantiate the agent
            model = PPO2(
                CustomCnnPolicy,
                env,
                policy_kwargs=policy_kwargs,
                ent_coef=ent_coef,
                n_steps=n_steps,
                nminibatches=nminibatches,
                noptepochs=noptepochs,
                learning_rate=learning_rate,
                verbose=1,
                tensorboard_log=folder_path + '/ppo_v2_tensorboard/')

            # model = PPO2(MlpPolicy, env, ent_coef=ent_coef, n_steps=n_steps, nminibatches=nminibatches, noptepochs=noptepochs, learning_rate=learning_rate, verbose=1, tensorboard_log=folder_path + '/ppo_v2_tensorboard/')
            # Train the agent
            model.learn(total_timesteps=rl_time_steps)  # this will crash explaining that the invalid value originated from the env

            model.save(folder_path + '/model')

            copyfile('gym_config/config.ini', folder_path + '/config.ini')

            # Evaluate model on 50 random environments of same parameters.
            self.evaluate_policy(folder_path, num_evals, env, garden_x, garden_y, sector_width, sector_height, is_baseline=False)

            # Graph evaluations
            # self.graph_utils.graph_evaluations(folder_path, garden_x, garden_y, time_steps, step, num_evals, num_plant_types)

        profiler_object.disable()

        # dump the profiler stats 
        s = io.StringIO()
        ps = pstats.Stats(profiler_object, stream=s).sort_stats('cumulative')
        pathlib.Path(folder_path + '/Timings').mkdir(parents=True, exist_ok=True)
        ps.dump_stats(folder_path + '/Timings/dump.txt')

        # convert to human readable format
        out_stream = open(folder_path + '/Timings/time.txt', 'w')
        ps = pstats.Stats(folder_path + '/Timings/dump.txt', stream=out_stream)
        ps.strip_dirs().sort_stats('cumulative').print_stats()

    def batch_run(self, n, rl_config, garden_x, garden_y, sector_width, sector_height, num_plant_types, num_plants_per_type, policy_kwargs=[], num_evals=1, is_baseline=[], baseline_policy=None):
        assert(len(rl_config) == n)
        assert(len(garden_x) == n)
        assert(len(garden_y) == n)
        assert(len(sector_width) == n)
        assert(len(sector_height) == n)
        assert(len(num_plant_types) == n)
        assert(len(num_plants_per_type) == n)

        if is_baseline:
            assert(len(is_baseline) == n)
            assert(baseline_policy != None)
            assert(len(policy_kwargs) == n)
        else:
            assert(len(policy_kwargs) == 1)

        if is_baseline:
            for i in range(n):
                filename_time = time.strftime('%Y-%m-%d-%H-%M-%S')

                if is_baseline[i]:
                    folder_path = self.file_utils.createBaselineSingleRunFolder(
                        garden_x[i],
                        garden_y[i],
                        num_plant_types[i],
                        num_plants_per_type[i],
                        policy_kwargs[i],
                        filename_time) 
                    self.file_utils.create_config(
                        rl_time_steps=rl_config[i]['time_steps'],
                        garden_x=garden_x[i],
                        garden_y=garden_y[i],
                        sector_width=sector_width[i],
                        sector_height=sector_height[i],
                        num_plant_types=num_plant_types[i],
                        num_plants_per_type=num_plants_per_type[i],
                        cnn_args=policy_kwargs[i])
                    self.single_run(folder_path, num_evals, is_baseline=True, baseline_policy=baseline_policy)
                else:
                   folder_path = self.file_utils.createRLSingleRunFolder(
                       garden_x[i],
                       garden_y[i],
                       num_plant_types[i],
                       num_plants_per_type[i],
                       rl_config[i],
                       policy_kwargs[i],
                       filename_time)
                   self.file_utils.create_config(
                        rl_time_steps=rl_config[i]['time_steps'],
                        garden_x=garden_x[i],
                        garden_y=garden_y[i],
                        sector_width=sector_width[i],
                        sector_height=sector_height[i],
                        num_plant_types=num_plant_types[i],
                        num_plants_per_type=num_plants_per_type[i],
                        ent_coef=rl_config[i]['ent_coef'],
                        nminibatches=rl_config[i]['nminibatches'],
                        noptepochs=rl_config[i]['noptepochs'],
                        learning_rate=rl_config[i]['learning_rate'],
                        cnn_args=policy_kwargs[i])
                   self.single_run(folder_path, num_evals, policy_kwargs[i], is_baseline=False)
        else:
            for i in range(n):
                filename_time = time.strftime('%Y-%m-%d-%H-%M-%S')
                folder_path = self.file_utils.createRLSingleRunFolder(
                    garden_x[i],
                    garden_y[i],
                    num_plant_types[i],
                    num_plants_per_type[i],
                    rl_config[i],
                    policy_kwargs[i],
                    filename_time)
                self.file_utils.create_config(
                    rl_time_steps=rl_config[i]['time_steps'],
                    garden_x=garden_x[i],
                    garden_y=garden_y[i],
                    sector_width=sector_width[i],
                    sector_height=sector_height[i],
                    num_plant_types=num_plant_types[i],
                    num_plants_per_type=num_plants_per_type[i],
                    ent_coef=rl_config[i]['ent_coef'],
                    nminibatches=rl_config[i]['nminibatches'],
                    noptepochs=rl_config[i]['noptepochs'],
                    learning_rate=rl_config[i]['learning_rate'],
                    cnn_args=policy_kwargs[i])
                self.single_run(folder_path, num_evals, policy_kwargs[0], is_baseline=False)

if __name__ == '__main__':
    n = 1
    rl_config = [
        {
            'rl_algorithm': 'CNN', 
            'time_steps': 50, 
            'ent_coef': 0.0,
            'n_steps': 200,
            'nminibatches': 4,
            'noptepochs': 4,
            'learning_rate': 1e-4
        }    
    ]
    garden_x = [300]
    garden_y = [150]
    sector_width = [30]
    sector_height = [15]
    num_plant_types = [1]
    num_plants_per_type = [1]
    is_baseline = [False]
    import simulatorv2.baselines.baseline_policy as bp
    baseline_policy = bp.baseline_policy
    policy_kwargs = [
        {
            "OUTPUT_X": 2,
            "OUTPUT_Y": 2,
            "NUM_HIDDEN_LAYERS": 1,
            "NUM_FILTERS": 3, # 2k+1 for new state representation
            "NUM_CONVS": 1,
            "FILTER_SIZE": 1,
            "STRIDE": 1,
            'CC_COEF': 0,
            'WATER_COEF': 1
        }
    ]
    num_evals = 1
    Pipeline().batch_run(n, rl_config, garden_x, garden_y, sector_width, sector_height, num_plant_types, num_plants_per_type, num_evals=num_evals, policy_kwargs=policy_kwargs, baseline_policy=baseline_policy, is_baseline=is_baseline)
