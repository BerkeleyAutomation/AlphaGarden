#!/usr/bin/env python3
import gym
import simalphagarden
import time
import json
import pathlib
import configparser
import matplotlib.pyplot as plt
import numpy as np
from cnn_policy import CustomCnnPolicy
from shutil import copyfile
from simulatorv2 import SimAlphaGardenWrapper
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
import cProfile
import pstats
import io

class Pipeline:
    def __init__(self):
        pass

    def create_config(self, rl_time_steps=3000000, garden_time_steps=50, garden_x=10, garden_y=10, num_plant_types=2, num_plants_per_type=1, step=1, action_low=0.0, action_high=1.0, obs_low=0, obs_high=1000, ent_coef=0.01, n_steps=40000, nminibatches=4, noptepochs=4, learning_rate=1e-8, cnn_args=None):
        config = configparser.ConfigParser()
        config.add_section('rl')
        config['rl']['time_steps'] = str(rl_time_steps)
        config['rl']['ent_coef'] = str(ent_coef)
        config['rl']['n_steps'] = str(n_steps)
        config['rl']['nminibatches'] = str(nminibatches)
        config['rl']['noptepochs'] = str(noptepochs)
        config['rl']['learning_rate'] = str(learning_rate)
        if cnn_args:
            config.add_section('cnn')
            config['cnn']['output_x'] = str(cnn_args["OUTPUT_X"])
            config['cnn']['output_y'] = str(cnn_args["OUTPUT_Y"])
            config['cnn']['num_hidden_layers'] = str(cnn_args["NUM_HIDDEN_LAYERS"])
            config['cnn']['num_filters'] = str(cnn_args["NUM_FILTERS"])
            config['cnn']['num_convs'] = str(cnn_args["NUM_CONVS"])
            config['cnn']['filter_size'] = str(cnn_args["FILTER_SIZE"])
            config['cnn']['stride'] = str(cnn_args["STRIDE"])
            config['cnn']['cc_coef'] = str(cnn_args['CC_COEF'])
            config['cnn']['water_coef'] = str(cnn_args['WATER_COEF'])
        config.add_section('garden')
        config['garden']['time_steps'] = str(garden_time_steps)
        config['garden']['X'] = str(garden_x)
        config['garden']['Y'] = str(garden_y)
        config['garden']['num_plant_types'] = str(num_plant_types)
        config['garden']['num_plants_per_type'] = str(num_plants_per_type)
        config['garden']['step'] = str(step)
        config.add_section('action')
        config['action']['low'] = str(action_low)
        config['action']['high'] = str(action_high)
        config.add_section('obs')
        config['obs']['low'] = str(obs_low)
        config['obs']['high'] = str(obs_high)

        pathlib.Path('gym_config').mkdir(parents=True, exist_ok=True)
        with open('gym_config/config.ini', 'w') as configfile:
            config.write(configfile)

    def running_avg(self, list1, list2, i):
        return [(x * i + y) / (i + 1) for x,y in zip(list1, list2)]

    def plot_water_map(self, folder_path, i, actions, m, n, plants):
        fig = plt.figure(figsize=(10, 10))
        heatmap = np.sum(actions, axis=0)
        heatmap = heatmap.reshape((m, n))
        plt.imshow(heatmap, cmap='Blues', origin='lower', interpolation='nearest')
        for plant in plants:
            plt.plot(plant[0], plant[1], marker='X', markersize=20, color="lawngreen")
        pathlib.Path(folder_path + '/Graphs').mkdir(parents=True, exist_ok=True)
        plt.savefig('./' + folder_path + '/Graphs/water_map_' + str(i) + '.png')

    def plot_final_garden(self, folder_path, i, garden, x, y, step):
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.xlim((0, x*step))
        plt.ylim((0, y*step))
        ax.set_aspect('equal')

        major_ticks = np.arange(0, x * step, 1) 
        minor_ticks = np.arange(0, x * step, step)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        rows = garden.shape[0]
        cols = garden.shape[1]
        plant_locations = []
        for x in range(0, rows):
            for y in range(0, cols):
                if garden[x,y] != 0:
                    plant_locations.append((x, y))
                    circle = plt.Circle((x*step,y*step), garden[x,y], color="green", alpha=0.4)
                    plt.plot(x*step, y*step, marker='X', markersize=15, color="lawngreen")
                    ax.add_artist(circle)
        pathlib.Path(folder_path + '/Graphs').mkdir(parents=True, exist_ok=True)
        plt.savefig('./' + folder_path + '/Graphs/final_garden_' + str(i) + '.png')
        return plant_locations

    def plot_average_reward(self, folder_path, reward, days, x_range, y_range, ticks):
        fig = plt.figure(figsize=(28, 10))
        plt.xticks(np.arange(0, days + 5, 5))
        plt.yticks(np.arange(x_range, y_range, ticks))
        plt.title('Average Reward Over ' + str(days) + ' Days', fontsize=18)
        plt.xlabel('Day', fontsize=16)
        plt.ylabel('Reward', fontsize=16)

        plt.plot([i for i in range(days)], reward, linestyle='--', marker='o', color='g')
        pathlib.Path(folder_path + '/Graphs').mkdir(parents=True, exist_ok=True)
        plt.savefig('./' + folder_path + '/Graphs/avg_reward.png')

    def plot_stddev_reward(self, folder_path, garden_time_steps, reward, reward_stddev, days, x_range, y_range, ticks):
        fig = plt.figure(figsize=(28, 10))
        plt.xticks(np.arange(0, days, 10))
        plt.yticks(np.arange(x_range, y_range, ticks))
        plt.title('Std Dev of Reward Over ' + str(days) + ' Days', fontsize=18)
        plt.xlabel('Day', fontsize=16)
        plt.ylabel('Reward', fontsize=16)

        plt.errorbar([i for i in range(garden_time_steps)], reward, reward_stddev, linestyle='None', marker='o', color='g')
        pathlib.Path(folder_path + '/Graphs').mkdir(parents=True, exist_ok=True)
        plt.savefig('./' + folder_path + '/Graphs/std_reward.png')

    def graph_evaluations(self, folder_path, garden_time_steps, garden_x, garden_y, time_steps, step, num_evals, num_plant_types):
        obs = [0] * time_steps
        r = [0] * time_steps
        for i in range(num_evals):
            with open(folder_path + '/Returns/predict_' + str(i) + '.json') as f_in:
                data = json.load(f_in)
                obs = data['obs']
                rewards = data['rewards']
                r = self.running_avg(r, rewards, i)
                action = data['action']

                final_obs = obs[time_steps-2]
                dimensions = len(final_obs)
                garden = np.array([[0.0 for x in range(dimensions)] for y in range(dimensions)])
                for x in range(dimensions):
                    s = np.array([0.0 for d in range(dimensions)])
                    s = np.add(s, np.array(final_obs[x]).T[-2])
                    garden[x] = s

                plant_locations = self.plot_final_garden(folder_path, i, garden, garden_x, garden_y, step)
                self.plot_water_map(folder_path, i, action, garden_x, garden_y, plant_locations)

        rewards_stddev = [np.std(val) for val in r]

        min_r = min(r) - 10
        max_r = max(r) + 10
        self.plot_average_reward(folder_path, r, time_steps, min_r, max_r, abs(min_r - max_r) / 10)
        self.plot_stddev_reward(folder_path, garden_time_steps, rewards, rewards_stddev, time_steps, min_r, max_r, abs(min_r - max_r) / 10)

    def evaluate_policy(self, folder_path, num_evals, env, garden_x, garden_y, is_baseline=False, baseline_policy=None, step=1):
        model = None
        if not is_baseline:
            model = PPO2.load('./' + folder_path + '/model')
        done = False
        for i in range(num_evals):
            obs = env.reset()
            garden_obs = env.env_method('get_garden_state')
            e = {'obs_avg_action': [], 'obs_action': [], 'obs': [], 'rewards': [], 'action': []}

            obs_avg_action = {}
            for x in range(garden_x):
                for y in range(garden_y):
                    obs_avg_action[x, y] = 0

            step_counter = 0

            while not done:
                action = None
                if is_baseline:
                    action = baseline_policy(garden_obs, step, 0.5, 0.5, 5)
                else:
                    action, _states = model.predict(obs)
                obs, rewards, done, _ = env.step(action)
                garden_obs = env.env_method('get_garden_state')
                radius_grid = env.env_method('get_radius_grid')
                
                if not done:
                    step_counter = env.env_method('get_current_step')[0]

                    rg_list = radius_grid[0].tolist()
                    obs_action_pairs = []
                    for x in range(garden_x):
                        for y in range(garden_y):
                            cell = (x, y)
                            cell_action = action[0][x * garden_x + y]
                            obs_action_pairs.append({str(cell) : (str(rg_list[x][y][0]), str(cell_action))})
                            obs_avg_action[cell] += cell_action
                    e['obs_action'].append({step_counter : obs_action_pairs})

                    e['obs'].append(garden_obs[0].tolist())
                    e['rewards'].append(rewards.item())
                    e['action'].append(action[0].tolist())
                    env.render()
            done = False

            for x in range(garden_x):
                for y in range(garden_y):
                    obs_avg_action[(x, y)] /= step_counter
                    e['obs_avg_action'].append({str((x, y)) : obs_avg_action[(x, y)], 'final': rg_list[x][y][0]})
            
            env.env_method('show_animation')

            pathlib.Path(folder_path + '/Returns').mkdir(parents=True, exist_ok=True)
            filename = folder_path + '/Returns' + '/predict_' + str(i) + '.json'
            f = open(filename, 'w')
            f.write(json.dumps(e, indent=4))
            f.close()

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
        garden_time_steps = config.getint('garden', 'time_steps')
        garden_x = config.getint('garden', 'X')
        garden_y = config.getint('garden', 'Y')
        # Z axis contains a matrix for every plant type plus one for water levels.
        garden_z = 2 * config.getint('garden', 'num_plant_types') + 1 
        action_low = config.getfloat('action', 'low')
        action_high = config.getfloat('action', 'high')
        obs_low = config.getint('obs', 'low')
        obs_high = config.getint('obs', 'high')

        env = gym.make(
                    'simalphagarden-v0',
                    wrapper_env=SimAlphaGardenWrapper(time_steps, garden_x, garden_y, num_plant_types, num_plants_per_type, step=step),
                    garden_x=garden_x,
                    garden_y=garden_y,
                    garden_z=garden_z,
                    action_low=action_low,
                    action_high=action_high,
                    obs_low=obs_low,
                    obs_high=obs_high)
        env = DummyVecEnv([lambda: env])
        env = VecCheckNan(env, raise_exception=False)

        if is_baseline:
            copyfile('gym_config/config.ini', folder_path + '/config.ini')

            # Evaluate baseline on 50 random environments of same parameters.
            self.evaluate_policy(folder_path, num_evals, env, garden_x, garden_y, is_baseline=True, baseline_policy=baseline_policy, step=1)

            # Graph evaluations
            self.graph_evaluations(folder_path, garden_time_steps, garden_x, garden_y, time_steps, step, num_evals, num_plant_types)
        else:
            pathlib.Path(folder_path + '/ppo_v2_tensorboard').mkdir(parents=True, exist_ok=True)
            # Instantiate the agent
#            model = PPO2(CustomCnnPolicy, env, policy_kwargs=policy_kwargs, ent_coef=ent_coef, n_steps=n_steps, nminibatches=nminibatches, noptepochs=noptepochs, learning_rate=learning_rate, verbose=1, tensorboard_log=folder_path + '/ppo_v2_tensorboard/')

            model = PPO2(MlpPolicy, env, ent_coef=ent_coef, n_steps=n_steps, nminibatches=nminibatches, noptepochs=noptepochs, learning_rate=learning_rate, verbose=1, tensorboard_log=folder_path + '/ppo_v2_tensorboard/')
            # Train the agent
            model.learn(total_timesteps=rl_time_steps)  # this will crash explaining that the invalid value originated from the env

            model.save(folder_path + '/model')

            copyfile('gym_config/config.ini', folder_path + '/config.ini')

            # Evaluate model on 50 random environments of same parameters.
            self.evaluate_policy(folder_path, num_evals, env, garden_x, garden_y, is_baseline=False)

            # Graph evaluations
            self.graph_evaluations(folder_path, garden_time_steps, garden_x, garden_y, time_steps, step, num_evals, num_plant_types)

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

    def createRLSingleRunFolder(self, garden_x, garden_y, num_plant_types, num_plants_per_type, rl, policy_kwargs, time):
        parent_folder = rl['rl_algorithm']
        pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
        sub_folder = parent_folder + '/' + str(garden_x) + 'x' + str(garden_y) + '_garden_' + str(num_plant_types*num_plants_per_type) + '_plants_' + str(rl['time_steps']) + '_timesteps_' + str(rl['learning_rate']) + '_learningrate_' + str(rl['n_steps']) + '_batchsize_' + str(policy_kwargs['CC_COEF']) + '_cropcoef_' + str(policy_kwargs['WATER_COEF']) + '_watercoef_' + time 
        pathlib.Path(sub_folder).mkdir(parents=True, exist_ok=False)
        return sub_folder

    def createBaselineSingleRunFolder(self, garden_x, garden_y, num_plant_types, num_plants_per_type, policy_kwargs, time):
        parent_folder = 'Baseline' 
        pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
        sub_folder = parent_folder + '/' + str(garden_x) + 'x' + str(garden_y) + '_garden_' + str(num_plant_types*num_plants_per_type) + '_plants_' + str(policy_kwargs['CC_COEF']) + '_cropcoef_' + str(policy_kwargs['WATER_COEF']) + '_watercoef_' + time 
        pathlib.Path(sub_folder).mkdir(parents=True, exist_ok=False)
        return sub_folder

        profiler_object.disable()

        # dump the profiler stats 
        s = io.StringIO()
        ps = pstats.Stats(profiler_object, stream=s).sort_stats('cumulative')
        pathlib.Path('Timings').mkdir(parents=True, exist_ok=True)
        ps.dump_stats('Timings/dump.txt')

        # convert to human readable format
        out_stream = open('Timings/time.txt', 'w')
        ps = pstats.Stats('Timings/dump.txt', stream=out_stream)
        ps.strip_dirs().sort_stats('cumulative').print_stats()
        filename_time = str(filename_time)


    def batch_run(self, n, rl_config, garden_x, garden_y, num_plant_types, num_plants_per_type, policy_kwargs=[], num_evals=1, is_baseline=[], baseline_policy=None):
        assert(len(rl_config) == n)
        assert(len(garden_x) == n)
        assert(len(garden_y) == n)
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
                    folder_path = self.createBaselineSingleRunFolder(garden_x[i], garden_y[i], num_plant_types[i], num_plants_per_type[i], policy_kwargs[i], filename_time) 
                    self.create_config(rl_time_steps=rl_config[i]['time_steps'], garden_x=garden_x[i], garden_y=garden_y[i], num_plant_types=num_plant_types[i], num_plants_per_type=num_plants_per_type[i], cnn_args=policy_kwargs[i])
                    self.single_run(folder_path, num_evals, is_baseline=True, baseline_policy=baseline_policy)
                else:
                   folder_path = self.createRLSingleRunFolder(garden_x[i], garden_y[i], num_plant_types[i], num_plants_per_type[i], rl_config[i], policy_kwargs[i], filename_time)
                   self.create_config(\
                        rl_time_steps=rl_config[i]['time_steps'], garden_x=garden_x[i], garden_y=garden_y[i], num_plant_types=num_plant_types[i], num_plants_per_type=num_plants_per_type[i], ent_coef=rl_config[i]['ent_coef'], nminibatches=rl_config[i]['nminibatches'], noptepochs=rl_config[i]['noptepochs'], learning_rate=rl_config[i]['learning_rate'], cnn_args=policy_kwargs[i])
                   self.single_run(folder_path, num_evals, policy_kwargs[i], is_baseline=False)
        else:
            for i in range(n):
                filename_time = time.strftime('%Y-%m-%d-%H-%M-%S')
                folder_path = self.createRLSingleRunFolder(garden_x[i], garden_y[i], num_plant_types[i], num_plants_per_type[i], rl_config[i], policy_kwargs[i], filename_time)
                self.create_config(\
                    rl_time_steps=rl_config[i]['time_steps'], garden_x=garden_x[i], garden_y=garden_y[i], num_plant_types=num_plant_types[i], num_plants_per_type=num_plants_per_type[i], ent_coef=rl_config[i]['ent_coef'], nminibatches=rl_config[i]['nminibatches'], noptepochs=rl_config[i]['noptepochs'], learning_rate=rl_config[i]['learning_rate'], cnn_args=policy_kwargs[i])
                self.single_run(folder_path, num_evals, policy_kwargs[0], is_baseline=False)

if __name__ == '__main__':
    n = 1
    rl_config = [
        {
            'rl_algorithm': 'MLP', 
            'time_steps': 40,
            'ent_coef': 0.0,
            'n_steps': 40000,
            'nminibatches': 4,
            'noptepochs': 4,
            'learning_rate': 1e-2
        }
    ]
    garden_x = [50]
    garden_y = [50]
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
            'CC_COEF': 10,
            'WATER_COEF': 100
        }
    ]
    num_evals = 1
    Pipeline().batch_run(n, rl_config, garden_x, garden_y, num_plant_types, num_plants_per_type, num_evals=num_evals, policy_kwargs=policy_kwargs, baseline_policy=baseline_policy, is_baseline=is_baseline)
