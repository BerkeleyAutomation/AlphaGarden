import pickle
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, default='./', help='Location of saved pickle files of garden runs.')
args = parser.parse_args()

num_files = 0
no_action, irrigate, prune, irrigate_prune = 0, 0, 0, 0
for file in os.listdir(args.path):
    if file.endswith(".pkl"):
        num_files += 1
        _, _, _, actions = pickle.load(open(args.path + '/' + file, 'rb'))
        for day in actions:
            for a in day:
                if a == 0:
                    no_action += 1
                elif a == 1:
                    irrigate += 1
                elif a == 2:
                    prune += 1
                elif a == 3:
                    irrigate_prune += 1
no_action_avg = no_action / num_files
irrigate_avg = irrigate / num_files
prune_avg = prune / num_files
irrigate_prune_avg = irrigate_prune / num_files
total = no_action + irrigate + prune + irrigate_prune
print('no action:', no_action_avg, 'percent:', no_action / total)
print('irrigate:', irrigate_avg, 'percent:', irrigate / total)
print('prune:', prune_avg, 'percent:', prune / total)
print('irrigate+prune:', irrigate_prune_avg, 'percent:', irrigate_prune / total)