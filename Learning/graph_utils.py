import numpy as np
import json
import pathlib
import matplotlib.pyplot as plt

class GraphUtils:
    def __init__(self):
        pass
    
    def running_avg(self, list1, list2, i):
            return [(x * i + y) / (i + 1) for x,y in zip(list1, list2)]

    def plot_water_map(self, folder_path, i, actions, m, n, plants):
        plt.figure(figsize=(10, 10))
        heatmap = np.sum(actions, axis=0)
        heatmap = heatmap.reshape((m, n))
        plt.imshow(heatmap, cmap='Blues', origin='lower', interpolation='nearest')
        for plant in plants:
            plt.plot(plant[0], plant[1], marker='X', markersize=20, color="lawngreen")
        pathlib.Path(folder_path + '/Graphs').mkdir(parents=True, exist_ok=True)
        plt.savefig('./' + folder_path + '/Graphs/water_map_' + str(i) + '.png')

    def plot_final_garden(self, folder_path, i, garden, x, y, step):
        _, ax = plt.subplots(figsize=(10, 10))
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
        plt.figure(figsize=(28, 10))
        plt.xticks(np.arange(0, days + 5, 5))
        plt.yticks(np.arange(x_range, y_range, ticks))
        plt.title('Average Reward Over ' + str(days) + ' Days', fontsize=18)
        plt.xlabel('Day', fontsize=16)
        plt.ylabel('Reward', fontsize=16)

        plt.plot([i for i in range(days)], reward, linestyle='--', marker='o', color='g')
        pathlib.Path(folder_path + '/Graphs').mkdir(parents=True, exist_ok=True)
        plt.savefig('./' + folder_path + '/Graphs/avg_reward.png')

    def plot_stddev_reward(self, folder_path, reward, reward_stddev, days, x_range, y_range, ticks):
        plt.figure(figsize=(28, 10))
        plt.xticks(np.arange(0, days, 10))
        plt.yticks(np.arange(x_range, y_range, ticks))
        plt.title('Std Dev of Reward Over ' + str(days) + ' Days', fontsize=18)
        plt.xlabel('Day', fontsize=16)
        plt.ylabel('Reward', fontsize=16)

        plt.errorbar([i for i in range(days)], reward, reward_stddev, linestyle='None', marker='o', color='g')
        pathlib.Path(folder_path + '/Graphs').mkdir(parents=True, exist_ok=True)
        plt.savefig('./' + folder_path + '/Graphs/std_reward.png')

    def graph_evaluations(self, folder_path, garden_x, garden_y, time_steps, step, num_evals, num_plant_types):
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
        self.plot_stddev_reward(folder_path, rewards, rewards_stddev, time_steps, min_r, max_r, abs(min_r - max_r) / 10)