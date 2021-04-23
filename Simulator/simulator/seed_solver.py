import numpy as np
import os
# from helper import *
import copy
from plant_presets import *
import pickle

MAX_DISTANCE_FOR_GEEEDY_SWITCH = 25

def compute_scores(seed_locs, labels, NUM_SEEDS):

    R = np.zeros([NUM_SEEDS, NUM_SEEDS])
    for i, v1 in enumerate(labels):
        for j, v2 in enumerate(labels):
            R[i,j] = PLANTS_RELATION[PLANTS[v1]][PLANTS[v2]]

    distances = np.sqrt((np.transpose(seed_locs[:,0:1].repeat(NUM_SEEDS,1)) - seed_locs[:,0:1]) ** 2 + (np.transpose(seed_locs[:,1:2].repeat(NUM_SEEDS,1)) - seed_locs[:,1:2]) ** 2)
    diag = (1.0 - np.eye(NUM_SEEDS))
    A = (- (R / (distances**2+1e-5))) * diag
    return np.sum(A)

def compute_distance(a, b):
    distance = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return distance

def find_labels(NUM_SEEDS, labels, t_id, rres=False):

    ROW = int(np.sqrt(NUM_SEEDS))
    COL = int(np.ceil(NUM_SEEDS/ROW))

    ordered_points = []
    for i in range(ROW):
        for j in range(COL):
            ordered_points.append([i*90/(ROW-1)+5, j*90/(COL-1)+5])
            if len(ordered_points) == len(labels):
                break
        if len(ordered_points) == len(labels):
            break
    points = np.asarray(ordered_points)

    init_score = compute_scores(points, labels, NUM_SEEDS)

    label_hist = []
    trial = 0
    c = 0
    while True:
        trial += 1
        res = np.ones([len(labels), len(labels)]) * 1e+5
        for ii in range(len(labels)):
            for jj in range(len(labels)):
                if ii < jj:

                    if compute_distance(points[ii], points[jj]) < MAX_DISTANCE_FOR_GEEEDY_SWITCH:

                        new_labels = copy.copy(labels)
                        new_labels[ii] = labels[jj]
                        new_labels[jj] = labels[ii]

                        res[ii, jj] = compute_scores(points, new_labels, NUM_SEEDS)

        mini = np.argmin(res)
        a = mini // len(labels)
        b = mini % len(labels)
        new_labels = copy.copy(labels)
        new_labels[a] = labels[b]
        new_labels[b] = labels[a]
        labels = copy.copy(new_labels)
        label_hist.append(labels)

        current_score = compute_scores(points, labels, NUM_SEEDS)
        if trial % 3 == 0:
            print(trial, init_score, current_score)
            with open("data_img/trial_init" + str(trial) + "id" + str(t_id), "wb") as f:
                pickle.dump([points, labels], f)

        if current_score >= init_score:
            c += 1
        else:
            init_score = current_score
            c = 0
        if c > 2:
            break

    if rres:
        return [points, labels, init_score, label_hist]
    return [points, labels, init_score]
