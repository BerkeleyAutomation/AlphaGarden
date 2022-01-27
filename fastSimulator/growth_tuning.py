import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import time
import json
#from gym_fastag.envs.plant_presets import create_plant_data
import yaml

import gym
import gym_fastag


def save_radii(env):
    """
    For growth analysis.
    """
    seen = {}
    for i in range(len(env.plants.common_names)):
        name = env.plants.common_names[i]
        radius = env.plants.current_outer_radii[i]
        # print(name, radius, self.current_day)
        # For naming convention
        if name not in seen.keys():
            seen[name] = 1
        else:
            seen[name] += 1
        # For saving files
        if env.current_day == 1:
            past_radi = [radius]
            pkl.dump(past_radi, open('./data/sim_growth_data/' + name + str(seen[name]) + '.p', 'wb'))
        else:
            past_radi = pkl.load(open('./data/sim_growth_data/' + name + str(seen[name]) + '.p', 'rb'))
            past_radi.append(radius)
            pkl.dump(past_radi, open('./data/sim_growth_data/' + name + str(seen[name]) + '.p', 'wb'))


def Convert(string): 
    li = list(string.split(", ")) 
    for i in range(len(li)):
        li[i] = float(li[i])
    return li

def plotBothR(x, y, z, file_name):
    path = './data/growth_plots/'
    plt.figure()
    plt.plot(x, y, 'b', label="Real world plant")
    plt.plot(x, z, 'g', label="Simulated plant")
    plt.ylim(0,40)
    plt.xlim(1, 45)
    plt.title(file_name + " Radius over Time")
    plt.ylabel("Radius (cm)")
    plt.xlabel("Time (day)")
    plt.legend()
    fig1 = plt.gcf()
    # plt.show()
    fig1.savefig(path + file_name + '.png')
    plt.close()

def rmsd(a, b):
    ret = 0
    num = len(a)
    for i in range(num):
        sqrd = (a[i] - b[i])**2
        ret += sqrd
    ret /= num
    ret = np.sqrt(ret)
    return ret
    
def mae(a, b):
    ret = 0
    num = len(a)
    for i in range(num):
        sqrd = abs(a[i] - b[i])
        ret += sqrd
    ret /= num
    return ret

def create_average_real():
    ### For real world (OLD DATA)
    """
    Specify path for last data collected (A LOT MORE) or new data
    ALSO could be very interesting to seed how the new data compares to the old (you can plot this)
    """
    path = 'data/real_growth_data/' #OLD DATA

    file_list = os.listdir(path)
    list.sort(file_list)
    out = []
    for file in file_list:
        if file == '.DS_Store':
            continue
        f = open(path + file, "r")
        item = f.read()
        item = item[1:len(item)-1]
        l = Convert(item)
        out.append(l)

    #separate
    if '.DS_Store' == file_list[0]:
        file_list.pop(0)

    sorted_names = []
    sep_plants = []
    curr_plant = 'arug'
    curr = []
    for i in range(len(out)):
        if curr_plant not in sorted_names:
            # print("HERE: ", curr_plant)
            sorted_names.append(curr_plant)
        if curr_plant == file_list[i][:4]:
            curr.append(out[i])
        else:
            sep_plants.append(curr)
    #         print("LEN: ", len(curr))
            curr_plant = file_list[i][:4]
            curr = [out[i]]
    sep_plants.append(curr)

    #avg
    cc = 0
    count = 0
    sep_plants_avg = {}
    for a in sep_plants: 
        num = len(a) # number of text files
        div_num = 0 # number to divide by helper
        new = [] # average of all text files for one plant
        for i in range(len(a[0])): 
            total_sum = 0 #sum to be averaged for element i
            for n in range(num): 
                if a[n][i] == 0 and count > 20:
                    div_num += 1
                total_sum += a[n][i]
            if div_num >= num:
                total_sum = 0
            else:
                total_sum /= num - div_num
            div_num = 0
            count += 1
            new.append(total_sum)
        sep_plants_avg[sorted_names[cc]] = new
        cc += 1
    return sep_plants_avg

def create_average_sim():
    path = 'data/sim_growth_data/'
    file_list = os.listdir(path)
    list.sort(file_list)
    if file_list[0] == '.DS_Store':
        file_list.pop(0)
    avg = {}
    count = 0
    for i in np.arange(0, 8): #Dependent on seed placement/number of seeds np.arange(0, 24, 3)
        name = file_list[i][:4]
        print(name)
        data = pkl.load(open(path + file_list[i], 'rb'))
        avg[name] = data
    return avg

def plot_and_similarity(avg_real, avg_sim, r_max, water_use):
    plant_mae = {}
    days = np.arange(0,100)
    for key in avg_sim.keys():
        sim_radi = avg_sim[key]
        #MAE
        mae_diff = mae(avg_real[key][:38], sim_radi[:38])
        plant_mae[key] = mae_diff
        #PLOTTING
        real_padded = np.pad(avg_real[key], (0,len(sim_radi)-len(avg_real[key])), 'constant', constant_values=0)
        name = str(key) + 'PLANT_' + str(r_max) + 'RMAX_' + str(water_use) + 'WATERUSE'
        #print(sim_radi)
        plotBothR(days, real_padded[:100], sim_radi[:100], name)
    return plant_mae

def parameter_search():
    #load_bf = pkl.load(open('best_fit_data.p', 'rb'))
    best_fit = {} #if load_bf is None else load_bf
    best_plot = {}
    # water_use = 0.01
    # r_max = 55
    with open("gym_fastag/envs/config/tune.yaml") as setup_file:
        load_test_config = yaml.safe_load(setup_file)

    avg_real = create_average_real()
    for water_use in np.linspace(0.13, 0.21, 17):
        #for r_max in np.linspace(13, 55, 20):
            print("WATERUSE: ", water_use)
        #   print("RMAX: ", r_max)
        # Create new data
        # create_plant_data(key='TUNE', r_max=r_max, water_use=water_use)
        # Run sim

            #  load_test_config['water_use_efficiencies']['default_value']  = [water_use for _ in range(8)]
            load_test_config['water_use_efficiencies']['default_value'] = [water_use for _ in range(8)]
            radii = load_test_config['reference_outer_radii']['default_value'] # = [r_max for _ in range(8)]

            env = gym.make("fastag-v0", env_config=load_test_config)
            env.seed(0)
            obs = env.reset()
            past_radii = []
            names = env.plants.common_names
            past_radii.append(env.plants.current_outer_radii)
            for day in range(env.day_limit):
                action = env.baseline_policy(obs, day)
                action['prune'] = np.zeros_like(action['prune'])
                #if day % 20 == 0:
                #    env.render()
                obs, _, _, _ = env.step(action)
                past_radii.append(env.plants.current_outer_radii.tolist())
            past_radii = np.array(past_radii)
            plant_mae = {}
            #pp = ['Green Lettuce', 'Red_Lettuce', 'Borage', 'Swiss Chard', 'Kale', 'Cilantro', 'Radicchio', 'Turnip']
            me_hack = [35, 37, 35, 42, 32, 46, 34, 41]
            for i in range(len(names)):
                key = names[i][:4].lower()
                bound = me_hack[i]
                sim_radi = past_radii[:, i]
                r_max = radii[i]
                #water_use = water[i]
                # MAE
                mae_diff = mae(avg_real[key][:bound], sim_radi[:bound])
                plant_mae[key] = mae_diff
                # PLOTTING
                real_padded = np.pad(avg_real[key], (0, len(sim_radi) - len(avg_real[key])), 'constant', constant_values=0)
                name = names[i]  ##  str(key) + 'PLANT_' + str(r_max) + 'RMAX_' + str(water_use) + 'WATERUSE'
                # print(sim_radi)

                if key not in best_fit:
                    best_fit[key] = [mae_diff, (r_max, water_use)]
                    best_plot[key] = [real_padded.tolist(), sim_radi, name]
                elif mae_diff < best_fit[key][0]:
                    print("NEW BEST: ", key, mae_diff, r_max, water_use)
                    best_fit[key] = [mae_diff, (r_max, water_use)]
                    best_plot[key] = [real_padded.tolist(), sim_radi, name]

    days = np.arange(0, 100)
    for v in best_plot.values():
        plotBothR(days[:45], v[0][:45], v[1][:45], v[2])
    pkl.dump(best_fit, open('best_fit_data.p', 'wb'))
    return best_fit

if __name__ == '__main__':
    b_fit = parameter_search()
    print(b_fit)
    # sim = create_average_sim()
    # print(sim)


