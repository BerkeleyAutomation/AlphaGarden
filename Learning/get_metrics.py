#!/usr/bin/env python
import pickle
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, default='./', help='Location of saved pickle files of garden runs.')
args = parser.parse_args()

coverage, diversity, ent_1, ent_2 = [], [], [], []
coverage_list, diversity_list, water_use_list, entropy_1, entropy_2, = [], [], [], [], []
for file in os.listdir(args.path + '/'):
    if file.endswith(".pkl"):
        coverage, diversity, water_use, actions, ent_1, ent_2 = pickle.load(open(args.path + '/' + file, 'rb'))
        coverage_list.append(np.sum(coverage[20:51])/50)
        diversity_list.append(np.mean(diversity[20:51]))
        water_use_list.append(np.sum(water_use))
        entropy_1.append(np.mean(ent_1[20:51]))
        entropy_2.append(np.mean(ent_2[20:51]))
print('Average total coverage: ' + str(np.mean(coverage_list)))
print('Average diversity: ' + str(np.mean(diversity_list)))
print('Average MME-1: ' + str(np.mean(entropy_1)))
print('Average MME-2: ' + str(np.mean(entropy_2)))
print('Average total water use: ' + str(np.mean(water_use_list)))
