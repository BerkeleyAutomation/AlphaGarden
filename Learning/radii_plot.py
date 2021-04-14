#!/usr/bin/env python
import pickle
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, default='./', help='Location of saved pickle files of garden runs.')
parser.add_argument('--output', '-o', type=str, default='./', help='Location to save radii pngs.')
args = parser.parse_args()

radii = [0 for i in range(99)]
for file in os.listdir(args.path + '/'):
    if file.startswith("fast"):
    # if file.endswith(".pkl"):
        loaded = pickle.load(open(args.path + '/' + file, "rb"))
        radii = [a + b for a, b in zip(radii, loaded)]
radii = [r / 5 for r in radii]

radii2 = [0 for i in range(99)]
for file in os.listdir(args.path + '/'):
    if file.startswith("slow"):
    # if file.endswith(".pkl"):
        loaded = pickle.load(open(args.path + '/' + file, "rb"))
        radii2 = [a + b for a, b in zip(radii2, loaded)]
radii2 = [r / 5 for r in radii2]

fig, ax = plt.subplots()
ax.set_ylim([0, 25])
# fig.suptitle(file.split('_')[0] + ' Radius',  y=0.93)
fig.suptitle('Average Radius',  y=0.93)
plt.plot(radii, label='Fast Plants Avg Radius')
plt.plot(radii2, label='Slow Plants Avg Radius')
plt.legend()
# plt.savefig(args.output + '/' + file + '.png', bbox_inches='tight', pad_inches=0.02)
plt.savefig(args.output + '/' + 'fast_slow_avg' + '.png', bbox_inches='tight', pad_inches=0.02)
plt.clf()
plt.close()
