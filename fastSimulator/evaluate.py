import gym
import gym_fastag
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pandas as pd
import scipy.stats as st

# plants_16types_150x150garden_real_fixed_test.yaml
# plants_16types_150x150garden_invasive_fixed_test.yaml
# plants_3types_100x100garden_invasive_fixed_test.yaml
# plants_3types_100x100garden_real_fixed_test.yaml
with open("gym_fastag/envs/config/humanVsRobot.yaml") as setup_file:
    load_test_config = yaml.safe_load(setup_file)

env = gym.make("fastag-v0", env_config=load_test_config)
#env.seed(10000)

trail_rewards = []
trail_coverage = []
trail_water = []
trail_diversity = []

plant1 = []
plant2 = []
plant3 = []
plant4 = []

for trail in range(1):
    env.seed(10000 + trail)
    print('Trail ', trail)
    rewards = []
    obs = env.reset()

    for day in range(env.day_limit):
        print(day)
        action = env.baseline_policy_variable_irrigation(obs, day)
        # action = env.baseline_policy(obs, day)

        #render cc_image every 10 days
        # print(action)
        # if day % 10 == 0:
        #     env.render()

        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        ia = info['irr_amounts']
        #0-1 Green Lettuce
        #2-3 Red Lettuce
        #4-5 Borage
        #6-7 Swiss Chard
        #8-9 Kale
        #10-11 Cilantro
        #12-13 Radicchio
        #14-15 Turnip

        plant1.append(ia[2])
        plant2.append(ia[4]) 
        plant3.append(ia[8]) 
        plant4.append(ia[12]) 



        trail_water.append(info['total_irr'])

    # print("SUM: ", np.sum(trail_water))
    # print("MEAN: ", np.mean(trail_water))

    avg = (np.array(plant1) + np.array(plant2)) / 2

    print("Germ: ", np.mean(avg[:7]), avg[:7])   # days 0:7
    print("G1: ", np.mean(avg[7:21]), avg[7:21])     # days 8:21
    print("G2: ", np.mean(avg[21:60]), avg[21:60])     # days 21:60


    fig = plt.figure()
    plt.plot(env.coverages, label='Coverages')
    plt.plot(env.diversities, label='Diversities')
    plt.plot(rewards, label ='Rewards')
    plt.axhline(max(env.coverages), color='r')
    plt.legend()
    plt.show()

    fig1 = plt.figure()
    plt.plot(plant1, label='Red Lettuce')
    plt.plot(plant2, label='Borage')
    plt.plot(plant3, label='Kale')
    plt.plot(plant4, label='Radicchio')
    plt.ylim([0, 0.7])
    plt.legend()
    plt.show()

    # trail_water.append(- np.sum(env.irrigation_amounts))
    # trail_water.append(info['total_irr'])
    # trail_coverage.append(np.sum(np.array(env.coverages)[20:70]) / 50)
    # trail_diversity.append(np.sum(np.array(env.diversities)[20:70]) / 50)
    # r = np.sum(rewards)
    # trail_rewards.append(r)

def print_help(trail_data: list, name:str):
    data_mean = np.mean(trail_data)
    data_std = np.std(trail_data)
    low, high = st.t.interval(alpha=0.99, df=len(trail_data) - 1, loc=data_mean,
                              scale=st.sem(trail_data))
    print(name, ': ', low, data_mean, high, data_std)

# print_help(trail_data=trail_water, name='water')
# print_help(trail_data=trail_coverage, name='coverage')
# print_help(trail_data=trail_diversity, name='diversity')
# print_help(trail_data=trail_rewards, name='rewards')

# # dictionary of lists
# trail_dict = {'water': trail_water, 'coverage': trail_coverage, 'diversity': trail_diversity, 'rewards': trail_rewards}
# df = pd.DataFrame(trail_dict)
# # saving the dataframe
# df.to_csv('data/experiment_data/base_three_plants_inv_trail_results.csv')

