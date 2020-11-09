import pickle
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

save_dir = os.getcwd()
coverage_list, diversity_list, water_use_list, coverage_diversity_list = [], [], [], []
for file in os.listdir(save_dir):
    if file.endswith(".pkl"):
        if "0.1" in file:
            print(pickle.load(open(save_dir + '/' + file, 'rb')))
            coverage0_1,_, water_use0_1, _ = pickle.load(open(save_dir + '/' + file, 'rb'))
            print(coverage0_1, water_use0_1)
            covwat0_1 = np.multiply(coverage0_1, water_use0_1)
            print(covwat0_1)
        if "0.01" in file:
            coverage0_01,_, water_use0_01, _ = pickle.load(open(save_dir + '/' + file, 'rb'))
            covwat0_01 = np.multiply(coverage0_01, water_use0_01)
        if "0.2" in file:
            print(pickle.load(open(save_dir + '/' + file, 'rb')))
            print(save_dir + '/' + file)
            coverage0_2,_, water_use0_2, _ = pickle.load(open(save_dir + '/' + file, 'rb'))
            covwat0_2 = np.multiply(coverage0_2, water_use0_2)
        if "0.0001" in file:
            coverage0_0001,_, water_use0_0001, _ = pickle.load(open(save_dir + '/' + file, 'rb'))
            covwat0_0001 = np.multiply(coverage0_0001, water_use0_0001)
        if "0.02" in file:
            coverage0_02,_, water_use0_02, _ = pickle.load(open(save_dir + '/' + file, 'rb'))
            covwat0_02 = np.multiply(coverage0_02, water_use0_02)
        if "0.0005" in file:
            coverage0_0005,_, water_use0_0005, _ = pickle.load(open(save_dir + '/' + file, 'rb'))
            covwat0_0005 = np.multiply(coverage0_0005, water_use0_0005)
        if "0.00005" in file:
            coverage0_00005,_, water_use0_00005, _ = pickle.load(open(save_dir + '/' + file, 'rb'))
            covwat0_00005 = np.multiply(coverage0_00005, water_use0_00005)

dirname = os.path.dirname(save_dir)
if not os.path.exists(dirname):
    os.makedirs(dirname)
fig, ax = plt.subplots()
ax.set_ylim([0, 1])
plt.plot(covwat0_1, label='0.1')
plt.plot(covwat0_01, label='0.01')
plt.plot(covwat0_2, label='0.2')
plt.plot(covwat0_0001, label='0.0001')
plt.plot(covwat0_02, label='0.02')
plt.plot(covwat0_0005, label='0.0005')
plt.plot(covwat0_00005, label='0.00005')
plt.legend()
plt.savefig(save_dir + 'coverage_and_water'  + '.png', bbox_inches='tight', pad_inches=0.02)
plt.clf()
plt.close()

