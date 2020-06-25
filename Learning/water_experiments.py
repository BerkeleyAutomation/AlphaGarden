import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools
import collections

directory_list = list()
file_name = 'data_1.pkl'

fig, ax = plt.subplots()
ax.set_ylim([0, 1])
plt.xlabel("Time [days]", fontsize=16)
plt.ylabel("Total Coverage", fontsize=16)
title = "Total Coverage"
plt.ylabel("Cumulative Water", fontsize=16)
title = "Cumulative Water Sum"
plt.title(title, fontsize=18)

cum_cc = []
max_coverage = []
exp_names = []
total_water = []
diversity_2070 = []
no_actions = []
water_actions = []
prune_actions = []
prune_water_actions = []
cum_sum_water = []
for root, dirs, files in os.walk("/Users/sebastianoehme/Desktop/water_exp2/", topdown=False):
    for dir_name in dirs:
        path_to_file = os.path.join(root, dir_name, file_name)
        experiment_name = dir_name[20:]  # experiment names: 'M..t..' or 'M..t...'

        #if experiment_name[:3] != 'M10':
        #    continue
        coverage, diversity, water_use, actions = pickle.load(open(path_to_file, 'rb'))
        label_text_c = 'Coverage ' + experiment_name
        #label_text_w = 'Water cumulative sum ' + experiment_name

        exp_names.append(experiment_name)

        cum_cc.append(np.sum(coverage))
        max_coverage.append(np.max(coverage))
        total_water.append(np.sum(water_use))
        diversity_2070.append(np.sum(diversity[20:70]))
        cum_water = np.cumsum(water_use)/100.0
        cum_sum_water.append(cum_water)

        action_counter = collections.Counter(itertools.chain(*actions))
        no_actions.append(action_counter[0])
        water_actions.append(action_counter[1])
        prune_actions.append(action_counter[2])
        prune_water_actions.append(action_counter[3])

        plt.plot(coverage, label=label_text_c)
        #plt.plot(cum_water, label=label_text_w)


plt.legend()
plt.savefig('/Users/sebastianoehme/Desktop/water_exp2/' + 'coverage_top5' + '.png', bbox_inches='tight', pad_inches=0.02)
plt.close(fig)