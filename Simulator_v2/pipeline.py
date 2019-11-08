#!/usr/bin/env python3
import gym
import simalphagarden
import time
import json
import pathlib
import configparser
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile
from plant import Plant
from plant_type import PlantType
from SimAlphaGardenWrapper import SimAlphaGardenWrapper
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines import PPO2

class Pipeline:
    def __init__(self):
        pass

    def create_config(self, rl_time_steps=5000000, garden_time_steps=40, garden_x=10, garden_y=10, num_plant_types=2, num_plants_per_type=1, step=1, spread=1, light_amt=1, action_low=0.0, action_high=0.5, obs_low=0, obs_high=1000):
        config = configparser.ConfigParser()
        config.add_section('rl')
        config['rl']['time_steps'] = str(rl_time_steps)
        config.add_section('garden')
        config['garden']['time_steps'] = str(garden_time_steps)
        config['garden']['X'] = str(garden_x)
        config['garden']['Y'] = str(garden_y)
        config['garden']['num_plant_types'] = str(num_plant_types)
        config['garden']['num_plants_per_type'] = str(num_plants_per_type)
        config['garden']['step'] = str(step)
        config['garden']['spread'] = str(spread)
        config['garden']['light_amt'] = str(light_amt)
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

    def plot_water_map(self, folder_prefix, model_name, i, actions, m, n, plants):
        fig = plt.figure(figsize=(10, 10))
        heatmap = np.sum(actions, axis=0)
        heatmap = heatmap.reshape((m, n))
        plt.imshow(heatmap, cmap='Blues', origin='lower', interpolation='nearest')
        for plant in plants:
            plt.plot(plant[0], plant[1], marker='X', markersize=20, color="lawngreen")
        pathlib.Path(folder_prefix + '_Graphs/' + model_name).mkdir(parents=True, exist_ok=True) 
        plt.savefig('./' + folder_prefix + '_Graphs/' + model_name + '/water_map_' + str(i) + '.png')
        
    def plot_final_garden(self, folder_prefix, model_name, i, garden, x, y, step):
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.xlim((0, x * step))
        plt.ylim((0, y * step))
        ax.set_aspect('equal')

        major_ticks = np.arange(0, x * step + 1, x // 5)
        minor_ticks = np.arange(0, x * step + 1, step)
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
                    circle = plt.Circle((x,y) * step, garden[x,y], color="green", alpha=0.4)
                    plt.plot(x, y, marker='X', markersize=15, color="lawngreen")
                    ax.add_artist(circle)
        pathlib.Path(folder_prefix + '_Graphs/' + model_name).mkdir(parents=True, exist_ok=True) 
        plt.savefig('./' + folder_prefix + '_Graphs/' + model_name + '/final_garden_' + str(i) + '.png')
        return plant_locations
        
    def plot_average_reward(self, folder_prefix, model_name, reward, days, y_range):
        fig = plt.figure(figsize=(28, 10))
        plt.xticks(np.arange(0, days, 5))
        plt.yticks(np.arange(0.0, y_range, 1))
        plt.title('Average Reward Over ' + str(days) + ' Days', fontsize=18)
        plt.xlabel('Day', fontsize=16)
        plt.ylabel('Reward', fontsize=16)

        plt.plot([i for i in range(days)], reward, linestyle='--', marker='o', color='g')
        pathlib.Path(folder_prefix + '_Graphs/' + model_name).mkdir(parents=True, exist_ok=True) 
        plt.savefig('./' + folder_prefix + '_Graphs/' + model_name + '/avg_reward.png')
        
    def plot_stddev_reward(self, folder_prefix, model_name, reward, reward_stddev, days, y_range):
        fig = plt.figure(figsize=(28, 10))
        plt.xticks(np.arange(0, days, 10))
        plt.yticks(np.arange(0, y_range, 1))
        plt.title('Std Dev of Reward Over ' + str(days) + ' Days', fontsize=18)
        plt.xlabel('Day', fontsize=16)
        plt.ylabel('Reward', fontsize=16)

        plt.errorbar([i for i in range(40)], reward, reward_stddev, linestyle='None', marker='o', color='g')
        pathlib.Path(folder_prefix + '_Graphs/' + model_name).mkdir(parents=True, exist_ok=True) 
        plt.savefig('./' + folder_prefix + '_Graphs/' + model_name + '/std_reward.png')

    def graph_evaluations(self, folder_prefix, model_name, garden_x, garden_y, time_steps, step, num_evals, num_plant_types):
        obs = [0] * 182
        r = [0] * 182
        for i in range(num_evals):
            with open(folder_prefix + '_Returns/' + model_name + '/predict_' + str(i) + '.json') as f_in:
                data = json.load(f_in)
                obs = data['obs']
                rewards = data['rewards']
                r = self.running_avg(r, rewards, i)
                action = data['action']

                final_obs = obs[38]
                dimensions = len(final_obs)
                garden = np.array([[0.0 for x in range(dimensions)] for y in range(dimensions)])
                for x in range(dimensions):
                    s = np.array([0.0 for d in range(dimensions)])
                    for t in range(num_plant_types):
                        s = np.add(s, np.array(final_obs[x]).T[t])
                    garden[x] = s
                    
                plant_locations = self.plot_final_garden(folder_prefix, model_name, i, garden, garden_x, garden_y, step)
                self.plot_water_map(folder_prefix, model_name, i, action, garden_x, garden_y, plant_locations)

        rewards_stddev = [np.std(val) for val in r]

        self.plot_average_reward(folder_prefix, model_name, r, time_steps, max(r) + 10)
        self.plot_stddev_reward(folder_prefix, model_name, rewards, rewards_stddev, time_steps, max(r) + 10)

    def evaulate_policy(self, folder_prefix, num_evals, env, model_name='', is_baseline=False, baseline_policy=None, step=1):
        model = None
        if not is_baseline:
            model = PPO2.load('./' + folder_prefix + '_Models/' + model_name)
        done = False
        for i in range(num_evals):
            obs = env.reset()
            e = {'obs': [], 'rewards': [], 'action': []}
            while not done:
                action = None
                if is_baseline:
                    action = baseline_policy(obs, step, 0.5, 0.5, 5)
                else:
                    action, _states = model.predict(obs)
                obs, rewards, done, _ = env.step(action)
                e['obs'].append(obs[0].tolist())
                e['rewards'].append(rewards.item())
                e['action'].append(action[0].tolist())
                env.render()
            done = False

            pathlib.Path(folder_prefix + '_Returns/' + model_name).mkdir(parents=True, exist_ok=True) 
            filename = folder_prefix + '_Returns/' + model_name + '/predict_' + str(i) + '.json'
            f = open(filename, 'w')
            f.write(json.dumps(e))
            f.close()

    def single_run(self, filename_time, num_evals, is_baseline=False, baseline_policy=None):
        filename_time = str(filename_time)

        config = configparser.ConfigParser()
        config.read('gym_config/config.ini')

        rl_time_steps = config.getint('rl', 'time_steps')
        time_steps = config.getint('garden', 'time_steps')
        step = config.getint('garden', 'step')
        spread = config.getint('garden', 'spread')
        light_amt = config.getint('garden', 'light_amt')
        num_plants_per_type = config.getint('garden', 'num_plants_per_type')
        num_plant_types = config.getint('garden', 'num_plant_types')
        garden_x = config.getint('garden', 'X')
        garden_y = config.getint('garden', 'Y')
        # Z axis contains a matrix for every plant type plus one for water levels.
        garden_z = config.getint('garden', 'num_plant_types') + 1
        action_low = config.getfloat('action', 'low')
        action_high = config.getfloat('action', 'high')
        obs_low = config.getint('obs', 'low')
        obs_high = config.getint('obs', 'high')

        env = gym.make( 
                    'simalphagarden-v0', 
                    wrapper_env=SimAlphaGardenWrapper(time_steps, garden_x, garden_y, num_plant_types, num_plants_per_type, step=step, spread=spread, light_amt=light_amt),
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
            model_name = 'baseline_v2_' + filename_time

            pathlib.Path('Baseline_Configs').mkdir(parents=True, exist_ok=True) 
            copyfile('gym_config/config.ini', './Baseline_Configs/' + model_name + '.ini')

            # Evaluate baseline on 50 random environments of same parameters.
            self.evaulate_policy('Baseline', model_name=model_name, num_evals=num_evals, env=env, is_baseline=True, baseline_policy=baseline_policy, step=1)

            # Graph evaluations
            self.graph_evaluations('Baseline', model_name, garden_x, garden_y, time_steps, step, num_evals, num_plant_types)
        else:
            pathlib.Path('ppo_v2_tensorboard').mkdir(parents=True, exist_ok=True)
            # Instantiate the agent
            model = PPO2(MlpPolicy, env, learning_rate=1e-8, verbose=1, tensorboard_log="./ppo_v2_tensorboard/")

            # Train the agent
            model.learn(total_timesteps=rl_time_steps)  # this will crash explaining that the invalid value originated from the env
            
            pathlib.Path('PPO_Models').mkdir(parents=True, exist_ok=True) 
            model_name = 'ppo2_v2_' + filename_time
            model.save('./PPO_Models/' + model_name)

            pathlib.Path('PPO_Configs').mkdir(parents=True, exist_ok=True) 
            copyfile('gym_config/config.ini', './PPO_Configs/' + model_name + '.ini')

            # Evaluate model on 50 random environments of same parameters.
            self.evaulate_policy('PPO', model_name=model_name, num_evals=num_evals, env=env, is_baseline=False)

            # Graph evaluations
            self.graph_evaluations('PPO', model_name, garden_x, garden_y, time_steps, step, num_evals, num_plant_types)

    def batch_run(self, n, rl_time_steps, garden_x, garden_y, num_plant_types, num_plants_per_type, num_evals=50, is_baseline=[], baseline_policy=None):
        assert(len(rl_time_steps) == n)
        assert(len(garden_x) == n)
        assert(len(garden_y) == n)
        assert(len(num_plant_types) == n)
        assert(len(num_plants_per_type) == n)

        if is_baseline:
            assert(len(is_baseline) == n)
            assert(baseline_policy != None)

        if is_baseline:
            for i in range(n):
                filename_time = time.strftime('%Y-%m-%d-%H-%M-%S')
                self.create_config(rl_time_steps=rl_time_steps[i], garden_x=garden_x[i], garden_y=garden_y[i], num_plant_types=num_plant_types[i], num_plants_per_type=num_plants_per_type[i])
                if is_baseline[i]:
                    self.single_run(filename_time, num_evals, is_baseline=True, baseline_policy=baseline_policy)
                else:
                    self.single_run(filename_time, num_evals, is_baseline=False)
        else:
            for i in range(n):
                filename_time = time.strftime('%Y-%m-%d-%H-%M-%S')
                self.create_config(rl_time_steps=rl_time_steps[i], garden_x=garden_x[i], garden_y=garden_y[i], num_plant_types=num_plant_types[i], num_plants_per_type=num_plants_per_type[i])
                self.single_run(filename_time, num_evals, is_baseline=False)

if __name__ == '__main__':
    n = 3
    rl_time_steps = [1, 1, 1]
    garden_x = [10, 20, 50]
    garden_y = [10, 20, 50]
    num_plant_types = [2, 2, 3]
    num_plants_per_type = [2, 2, 2]
    is_baseline = [False, True, False]
    import Baselines.baseline_policy as bp
    baseline_policy = bp.baseline_policy
    Pipeline().batch_run(n, rl_time_steps, garden_x, garden_y, num_plant_types, num_plants_per_type, num_evals=1, is_baseline=is_baseline, baseline_policy=baseline_policy)