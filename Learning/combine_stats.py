#!/usr/bin/env python
import pickle
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from simulator.sim_globals import NUM_IRR_ACTIONS
from metric_compare import calc_avg_irr_prune

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, default='./', help='Location of saved pickle files of garden runs.')
parser.add_argument('--output', '-o', type=str, default='./generated/', help='Output folder for generated graphs')
parser.add_argument('--percents', '-t', type = bool, default=False, help='Contains percents in pkl file')
args = parser.parse_args()

coverage_list, diversity_list, water_use_list, coverage_diversity_list, mme_1_list, mme_2_list, irr_list, prune_list = [], [], [], [], [], [], [],[]

subfolders = [1,'5_new',7,'12_new',16]

if __name__ == '__main__':
    for subfolder in subfolders:
        curr_path = os.path.join(args.path, str(subfolder))
        for file in os.listdir(curr_path):
            if file.endswith(".pkl"):
                if args.percents:
                    coverage, diversity, water_use, actions, mme_1, mme_2, percents  = pickle.load(open(os.path.join(curr_path,file), 'rb'))
                coverage_list.append(np.mean(coverage[20:71]))
                diversity_list.append(np.mean(diversity[20:71]))
                water_use_list.append(np.sum([w*len(a) for w, a in zip(water_use, actions)]))
                coverage_diversity_list.append(np.mean([diversity[i] * coverage[i] for i in range(20, 71)]))
                mme_1_list.append(np.mean(mme_1[20:71]))
                mme_2_list.append(np.mean(mme_2[20:71]))
                avg = calc_avg_irr_prune(actions, 0, 101)
                irr_list.append(avg[0])
                prune_list.append(avg[1])
    data = np.mean([np.sum(pl) for pl in irr_list]), np.mean([np.sum(pl) for pl in prune_list]),  np.mean([np.mean(pl) for pl in irr_list]), np.mean([np.mean(pl) for pl in prune_list])
    print('Average total coverage: ' + str(np.mean(coverage_list)))
    print('Average diversity: ' + str(np.mean(diversity_list)))
    print('Average MME1: ' + str(np.mean(mme_1_list)))
    print('Average MME2: ' + str(np.mean(mme_2_list)))
    print('Average total water use: ' + str(np.mean(water_use_list)))
    print(f'avg total irr: {data[0]}, avg irr per day: {data[2]}')
    print(f'avg total prune: {data[1]}, avg prune per day: {data[3]}')
    metrics = np.mean(coverage_list), np.mean(diversity_list), np.mean(water_use_list), np.mean(mme_1_list), np.mean(mme_2_list), *data
    with open(os.path.join(args.path , 'data_combined.pkl'), 'wb') as f:
       pickle.dump(metrics, f)