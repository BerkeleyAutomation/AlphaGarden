import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

dirpath = '/Users/yahavavigal/Projects/AlphaGarden/AlphaGardenSim/Experiments/'
# paths = ['no_prune_policy_data/data_4.pkl', 'fixed_policy_data/data_4.pkl', 'adaptive_policy_data/data_6.pkl', 'trained_policy_data/data_17.pkl']
# paths_inv = ['./no_prune_policy_data_inv/data_6.pkl', './fixed_policy_data_inv/data_9.pkl', './adaptive_policy_data_inv/data_2.pkl', './trained_policy_data_inv/data_2.pkl']
paths = ['no_prune_policy_data/', 'fixed_policy_data/', 'adaptive_policy_data/', 'trained_policy_data/']
paths_inv = ['./no_prune_policy_data_inv/', './fixed_policy_data_inv/', './adaptive_policy_data_inv/', './trained_policy_data_inv/']
titles = ['No Pruning Policy', 'Fixed Automation Policy', 'Adaptive Automation Policy', 'Learned Policy']
titles_inv = ['No Pruning Policy - Invasive Species', 'Fixed Automation Policy - Invasive Species', 'Adaptive Automation Policy - Invasive Species', 'Learned Policy - Invasive Species']
filename = ['No Pruning Policy', 'Fixed Policy', 'Adaptive Policy', 'Learned Policy']


def plot_coverag_and_diversity(path, title, output_filename):
    print(path)
    coverage, diversity, water_use, actions = pickle.load(open(path, 'rb'))
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1])
    ax.set_xlim(19,71)
    plt.xlabel("Time [days]", fontsize=16)
    plt.ylabel("Diversity / Total Coverage", fontsize=16)
    plt.title(title, fontsize=18)
    plt.plot(coverage, color='green', label="Coverage")
    plt.plot(diversity, color='orange', label="Diversity")
    min_div = round(min(diversity), 2)
    print(min_div)
    plt.legend(loc='center left')
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.02)
    # plt.show()
    plt.clf()
    plt.close()


def plot_water(paths, inv=False):
    coverage, diversity, water_use, actions = pickle.load(open(dirpath+paths[0], 'rb'))
    coverage, diversity, water_use2, actions = pickle.load(open(dirpath+paths[1], 'rb'))
    coverage, diversity, water_use3, actions = pickle.load(open(dirpath+paths[2], 'rb'))
    coverage, diversity, water_use4, actions = pickle.load(open(dirpath+paths[3], 'rb'))

    cum_water = np.cumsum(water_use)/100.0
    cum_water2 = np.cumsum(water_use2)/100.0
    cum_water3 = np.cumsum(water_use3)/100.0
    cum_water4 = np.cumsum(water_use4)/100.0
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1])
    plt.xlabel("Time [days]", fontsize=16)
    plt.ylabel("Cumulative Water", fontsize=16)
    title = "Cumulative Water Use - Invasive Species" if inv else "Cumulative Water Use"
    plt.title(title, fontsize=18)
    plt.plot(cum_water, color='blue', label="No Pruning")
    plt.plot(cum_water2, color='purple', label="Fixed Policy")
    plt.plot(cum_water3, color='red', label="Adaptive Policy")
    plt.plot(cum_water4, color='black', label="Learned Policy")
    min_div = round(min(diversity), 2)
    print(min_div)
    plt.legend(loc='center left')
    file_name = "Water_use_inv.pdf" if inv else "Water_use.pdf"
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.02)
    # plt.show()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='./')
    parser.add_argument('-f', '--filename', type=str, default='data_1.pkl')
    parser.add_argument('-t', '--test', type=str, default='coverage', help='coverage or water')
    parser.add_argument('-p', '--policy', type=int, default='0', help='[0|1|2|3] No pruning [0], Fixed [1], Adaptive [2], learned [3]')
    parser.add_argument('--invasive', action='store_true', help='With invasive species')
    args = parser.parse_args()

    inv = args.invasive
    dir_suff = args.dir
    folder = paths_inv[args.policy] if inv else paths[args.policy]
    path = dir_suff + folder + args.filename
    title = titles_inv[args.policy] if inv else titles[args.policy]
    suff = " invasive" if inv else ''
    output_filename = filename[args.policy] + suff + '.pdf'
    if args.test == 'coverage':
        plot_coverag_and_diversity(path, title, output_filename)
    else:
        if inv:
            plot_water(paths_inv, inv=True)
        else:
            plot_water(paths, inv=False)