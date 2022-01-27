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
with open("gym_fastag/envs/config/plants_3types_100x100garden_invasive_fixed_test.yaml") as setup_file:
    load_test_config = yaml.safe_load(setup_file)

env = gym.make("fastag-v0", env_config=load_test_config)
#env.seed(10000)

trail_rewards = []
trail_rewards_w = []
prune_dist = np.zeros(4)


trail_coverage = []
trail_water = []
trail_diversity = []
for trail in range(100):
    env.seed(10000 + trail)
    print('Trail ', trail)
    rewards = []
    rewards_w = []
    obs = env.reset()


    for day in range(env.day_limit):
        #action = env.action_space.sample()
        action = env.baseline_policy(obs, day)
        #render cc_image every 10 days
        #if day % 5 == 0:
        #   env.render()

        #prune_dist += action['prune']

        obs, reward, done, info = env.step(action)

        #print(info)

        rewards_w.append(np.maximum(reward - 0.1 * env.irrigation, 0.0))
        rewards.append(reward)
    #print(prune_dist)
    """fig = plt.figure()
    plt.plot(env.coverages, label='Coverages', color='green')
    plt.plot(env.diversities, ':', label='Diversities', color='orange')
    #plt.plot(env.irrigation_amounts, '--', label='Irrigation', color='blue')
    #plt.plot(rewards, '-.', label='Rewards', color='red')
    plt.xlim([0, 100])
    plt.ylim([0, 1])
    plt.ylabel(r'Coverage (%) / Diversity (%)')
    plt.xlabel("Time (day)")
    plt.title("Analytic Policy")
    # moving bottom spine up to y=0 position:
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().spines['bottom'].set_position(('data', 0))
    
    # turn off the right spine/ticks
    plt.gca().spines['right'].set_color('none')
    plt.gca().yaxis.tick_left()
    
    # turn off the top spine/ticks
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.tick_bottom()
    
    plt.ylabel(r'Coverage $\left[\frac{m^2}{m^2}\right]$ / Diversity H / Irriagtion $\left[\frac{m^3}{m^3}\right]$ / Reward r')
    plt.xlabel("Time (day)") 
    plt.legend()
    plt.tight_layout()
    plt.show()"""
    trail_water.append(- np.sum(env.irrigation_amounts))
    trail_coverage.append(np.sum(np.array(env.coverages)[20:70]) / 50)
    trail_diversity.append(np.sum(np.array(env.diversities)[20:70]) / 50)
    r = np.sum(rewards)
    r_w = np.sum(rewards_w)
    trail_rewards_w.append(r_w)
    trail_rewards.append(r)

def print_help(trail_data: list, name:str):
    data_mean = np.mean(trail_data)
    data_std = np.std(trail_data)
    low, high = st.t.interval(alpha=0.99, df=len(trail_data) - 1, loc=data_mean,
                              scale=st.sem(trail_data))
    #print(name, ': ', low, data_mean, high, data_std)
    print(name, ': ', str(round(data_mean, 2)) + ' \\pm ' + str(round(data_std, 2)))

print('MME ', np.average(trail_rewards), np.var(trail_rewards))
print('MME Water ', np.average(trail_rewards_w), np.var(trail_rewards_w))

print_help(trail_data=trail_water, name='water')
print_help(trail_data=trail_coverage, name='coverage')
print_help(trail_data=trail_diversity, name='diversity')
print_help(trail_data=trail_rewards, name='rewards')

# dictionary of lists
trail_dict = {'water': trail_water, 'coverage': trail_coverage, 'diversity': trail_diversity, 'rewards': trail_rewards}

df = pd.DataFrame(trail_dict)

# saving the dataframe
df.to_csv('data/experiment_data/base_three_plants_inv_trail_results.csv')