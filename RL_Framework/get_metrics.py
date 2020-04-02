#!/usr/bin/env python
import pickle
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, default='./', help='Location of saved pickle files of garden runs.')
args = parser.parse_args()

coverage_list, diversity_list, water_use_list = [], [], []
for file in os.listdir(args.path):
    if file.endswith(".pkl"):
        coverage, diversity, water_use, actions = pickle.load(open(args.path + '/' + file, 'rb'))
        coverage_list.append(np.sum(coverage))
        diversity_list.append(np.mean(diversity))
        water_use_list.append(np.sum(water_use))

print('Average total coverage: ' + str(np.mean(coverage_list)))
print('Average diversity: ' + str(np.mean(diversity_list)))
print('Average total water use: ' + str(np.mean(water_use_list)))
