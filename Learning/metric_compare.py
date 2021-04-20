#!/usr/bin/env python
import pickle
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from simulator.sim_globals import NUM_IRR_ACTIONS
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, default='./', help='Location of saved pickle files of garden runs.')
parser.add_argument('--graph', '-g',  type = bool, default=False, help='Generate Graphs in given folder')
parser.add_argument('--output', '-o', type=str, default='./generated/', help='Output folder for generated graphs')
parser.add_argument('--percents', '-t', type = bool, default=False, help='Contains percents in pkl file')
args = parser.parse_args()

def calc_avg_irr_prune(actions, low_thres, up_thres):
    prune = []
    irrigate = []
    for day in actions:
        unique, counts = np.unique(day, return_counts=True)
        dictionary = dict(zip(unique,counts))
        prune_total =0
        irr_total = 0
        for val in dictionary:
            if val == NUM_IRR_ACTIONS:
                irr_total += dictionary[val]
            elif val == NUM_IRR_ACTIONS+1:
                prune_total += dictionary[val]
            elif val == NUM_IRR_ACTIONS+2:
                prune_total += dictionary[val]
                irr_total += dictionary[val]
        prune.append(prune_total)
        irrigate.append(irr_total)
    return irrigate[low_thres:up_thres], prune[low_thres:up_thres]

colors = ['#925e78','#BD93BD','#F2EDEB','#F05365','#FABC2A']
def plot_compare(paths, vals, name, color = colors):
    fig, ax = plt.subplots()
    keys = list(paths.keys())
    b = ax.bar(keys, vals, color = colors)
    ax.set_title(name)
    ax.set_ylabel('Value')
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys)
    ax.set_xlabel("Method Used")
    ax.bar_label(b, fmt='%.2f', padding=3)
    return fig

def make_df(cols, data):
    return pd.DataFrame(data, columns = list(map(lambda c: c.replace('\n',' '),cols)))



coverage_list, diversity_list, water_use_list, coverage_diversity_list, mme_1_list, mme_2_list, irr_list, prune_list = [], [], [], [], [], [], [],[]
base_dir = '/mnt/c/Users/admin/Documents/git/ag/AlphaGarden/Learning/output/formal_experiments/'
paths = {'10\nRand\nPlants':'10','30\nRand\nPlants':'30','50\nRand\nPlants':'50','70\nRand\nPlants':'70','90\nRand\nPlants':'90','Plant\nCenters\nOnly':'100','110\nOriginal':'100_10random','Adaptive +\nPruning':'adaptive_np'}
keys = list(np.array(list(paths.keys()))[[6,0,1,2,3,4,5,7]])
paths = {k:paths[k] for k in keys}
for i, path in enumerate(list(paths.values())):
    for file in os.listdir(os.path.join(base_dir,path)):
        # print(file)
        if file.endswith(".pkl"):
            if args.percents:
                coverage, diversity, water_use, actions, mme_1, mme_2, percents  = pickle.load(open(os.path.join(base_dir, path,file), 'rb'))
            else:
                coverage, diversity, water_use, actions, mme_1, mme_2  = pickle.load(open(args.path + '/' + file, 'rb'))
            coverage_list.append(np.mean(coverage[20:71]))
            diversity_list.append(np.mean(diversity[20:71]))
            water_use_list.append(np.sum([w*len(a) for w, a in zip(water_use, actions)]))
            coverage_diversity_list.append(np.mean([diversity[i] * coverage[i] for i in range(20, 71)]))
            mme_1_list.append(np.mean(mme_1[20:71]))
            mme_2_list.append(np.mean(mme_2[20:71]))
            avg = calc_avg_irr_prune(actions, 0, 101)
            irr_list.append(avg[0])
            prune_list.append(avg[1])

# UNCOMMENT FOR all the comparison plots

# plot_compare(paths, coverage_list, "Coverage Averaged Days 20-70").savefig(os.path.join(base_dir, "coverage.png"))
# plot_compare(paths, diversity_list, "Diversity Averaged Days 20-70").savefig(os.path.join(base_dir, "diversity.png"))
# plot_compare(paths, water_use_list, "Total Water Use").savefig(os.path.join(base_dir, "wateruse.png"))
# # plot_compare(paths, coverage_diversity_list, "Coverage Diversity Averaged Dasy 20-70").savefig(os.path.join(base_dir, "wateruse.png"))
# plot_compare(paths, mme_1_list, "MME-1 Averaged Days 20-70").savefig(os.path.join(base_dir, "mme1.png"))
# plot_compare(paths, mme_2_list, "MME-2 Averaged Days 20-70").savefig(os.path.join(base_dir, "mme2.png"))
# plot_compare(paths, [np.sum(pl) for pl in irr_list], "Total Irrigation Actions Days 20-70").savefig(os.path.join(base_dir, "irr_tot.png"))
# plot_compare(paths, [np.sum(pl) for pl in prune_list], "Total Pruning Actions Days 20-70").savefig(os.path.join(base_dir, "prune_tot.png"))
# plot_compare(paths, [np.mean(pl) for pl in irr_list], "Irrigation Actions Averaged Days 20-70").savefig(os.path.join(base_dir, "irr.png"))
# plot_compare(paths, [np.mean(pl) for pl in prune_list], "Pruning Actions Averaged Days 20-70").savefig(os.path.join(base_dir, "prune.png"))
make_df(list(paths.keys()),[coverage_list, diversity_list, water_use_list, mme_1_list, mme_2_list, 
            [np.sum(pl) for pl in irr_list], [np.sum(pl) for pl in prune_list],  [np.mean(pl) for pl in irr_list], [np.mean(pl) for pl in prune_list]]).to_csv(os.path.join(base_dir, "df.csv"),index = False)