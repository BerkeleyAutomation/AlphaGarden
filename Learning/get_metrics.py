#!/usr/bin/env python
import pickle
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, default='./', help='Location of saved pickle files of garden runs.')
args = parser.parse_args()

coverage_list, diversity_list, water_use_list, coverage_diversity_list, global_diversity_list = [], [], [], [], []
for file in os.listdir(args.path):
    if file.endswith(".pkl"):
        coverage, diversity, water_use, actions, global_diversity = pickle.load(open(args.path + '/' + file, 'rb'))
        coverage_list.append(np.sum(coverage[20:71])/50)
        diversity_list.append(np.mean(diversity[20:71]))
        global_diversity_list.append(np.mean(global_diversity[20:71]))
        water_use_list.append(np.sum(water_use))
        coverage_diversity_list.append(np.mean([diversity[i] * coverage[i] for i in range(20, 71)]))

print('Average total coverage: ' + str(np.mean(coverage_list)))
print('Average diversity: ' + str(np.mean(diversity_list)))
print('Average global diversity with soil modification: ' + str(np.mean(global_diversity_list)))
print('Average total water use: ' + str(np.mean(water_use_list)))