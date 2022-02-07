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
with open("gym_fastag/envs/config/tune.yaml") as setup_file:
    load_test_config = yaml.safe_load(setup_file)

env = gym.make("fastag-v0", env_config=load_test_config)
#env.seed(10000)

trail_rewards = []
trail_coverage = []
trail_water = []
trail_diversity = []
for trail in range(1):
    env.seed(10000 + trail)
    print('Trail ', trail)
    rewards = []
    obs = env.reset()

    for day in range(env.day_limit):
        print(day)
        action = env.baseline_policy(obs, day)
        #render cc_image every 10 days
        #if day % 5 == 0:
        env.render(mode="save")
        obs, reward, done, info = env.step(action)

        rewards.append(reward)

    fig = plt.figure()
    plt.plot(env.coverages, label='Coverages')
    plt.plot(env.diversities, label='Diversities')
    plt.plot(rewards, label ='Rewards')
    plt.legend()
    plt.show()
    trail_water.append(- np.sum(env.irrigation_amounts))
    trail_coverage.append(np.sum(np.array(env.coverages)[20:70]) / 50)
    trail_diversity.append(np.sum(np.array(env.diversities)[20:70]) / 50)
    r = np.sum(rewards)
    trail_rewards.append(r)

def print_help(trail_data: list, name:str):
    data_mean = np.mean(trail_data)
    data_std = np.std(trail_data)
    low, high = st.t.interval(alpha=0.99, df=len(trail_data) - 1, loc=data_mean,
                              scale=st.sem(trail_data))
    print(name, ': ', low, data_mean, high, data_std)

# print_help(trail_data=trail_water, name='water')
print_help(trail_data=trail_coverage, name='coverage')
print_help(trail_data=trail_diversity, name='diversity')
print_help(trail_data=trail_rewards, name='rewards')

# # dictionary of lists
# trail_dict = {'water': trail_water, 'coverage': trail_coverage, 'diversity': trail_diversity, 'rewards': trail_rewards}
# df = pd.DataFrame(trail_dict)
# # saving the dataframe
# df.to_csv('data/experiment_data/base_three_plants_inv_trail_results.csv')

