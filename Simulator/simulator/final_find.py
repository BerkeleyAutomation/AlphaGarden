import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pickle
from scipy import optimize
from seed_solver import find_labels
import os
from plant_presets import *

# Parameters here
GARDEN_SIZE = 150
alphas = [0.5, 0.4, 0.3, 0.2, 0.1]
num_trials = 6

RATIO = 5
CONSTANT = 1.0
np.random.seed(10)

# The final placement result:
initial_try_id = 1
NUM_SEEDS = 100

# for NUM_SEEDS in [60]:
labels = np.asarray([i%len(PLANTS) for i in range(NUM_SEEDS)])
for initial_try_id in range(num_trials):
    if not os.path.exists("data/"+str(NUM_SEEDS)+"1data"):
        if not os.path.exists("data/"+str(NUM_SEEDS)+"1data-"+str(initial_try_id)):
            np.random.shuffle(labels)
            [points, labels, score] = find_labels(NUM_SEEDS=NUM_SEEDS, labels=labels, t_id=initial_try_id)
            with open("data/ordered", "wb") as f:
                pickle.dump([labels, points], f)
        # with open("data/"+str(NUM_SEEDS)+"data-"+str(initial_try_id), "rb") as f:
        #     [labels, points] = pickle.load(f)

        # [aa, bb] = np.random.choice(len(labels), 2)
        for OVERLAP_PERCENTAGE in alphas:
            seed_types = np.asarray(labels)
            seed_locs = np.asarray(points)
            PLANT_SIZE_TEMP = np.asarray([PLANT_SIZE[PLANTS[i]] for i in seed_types])
            tmp = PLANT_SIZE_TEMP.reshape([-1,1]).repeat(NUM_SEEDS, 1)
            r_max_sum = (tmp + tmp.transpose())

            R = np.zeros([NUM_SEEDS, NUM_SEEDS])
            for i, v1 in enumerate(seed_types):
                for j, v2 in enumerate(seed_types):
                    R[i,j] = PLANTS_RELATION[PLANTS[v1]][PLANTS[v2]]

            def func(x):
                seed_locs = x.reshape([-1, 2])
                distances = np.sqrt((np.transpose(seed_locs[:,0:1].repeat(NUM_SEEDS,1)) - seed_locs[:,0:1]) ** 2 + (np.transpose(seed_locs[:,1:2].repeat(NUM_SEEDS,1)) - seed_locs[:,1:2]) ** 2)
                diag = (1.0 - np.eye(NUM_SEEDS))
                # A = (- (R / (distances**2 + 1))) * diag
                A = (- (R / (distances**2 + 10))) * diag
                # return np.sum(A) / (NUM_SEEDS**2)
                return np.sum(A) * 0.1

            iss = func(seed_locs)

            x = seed_locs.reshape([-1])
            lb = np.asarray([[PLANT_SIZE[PLANTS[v]], PLANT_SIZE[PLANTS[v]]] for v in seed_types])
            lb = lb.reshape([-1])
            ub = GARDEN_SIZE - lb
            bounds = optimize.Bounds(lb=lb, ub=ub)

            def cons(x):
                seed_locs2 = x.reshape([-1, 2])
                distances2 = np.sqrt((np.transpose(seed_locs2[:,0:1].repeat(NUM_SEEDS,1)) - seed_locs2[:,0:1]) ** 2 + (np.transpose(seed_locs2[:,1:2].repeat(NUM_SEEDS,1)) - seed_locs2[:,1:2]) ** 2)
                a2 = (distances2 - r_max_sum * (1.0 - OVERLAP_PERCENTAGE)) + np.eye(NUM_SEEDS) * 1e+6
                a2 = a2.reshape([-1])
                return a2

            count = 0
            def cbf(Xi):
                global count
                if count % 10 == 0:
                    locations = Xi.reshape([-1, 2])
                    print(NUM_SEEDS, OVERLAP_PERCENTAGE, count, func(Xi))
                    with open('data_img/opt_trial' + str(count) + "trial" + str(initial_try_id) + "overlap" + str(OVERLAP_PERCENTAGE), "wb") as f:
                        pickle.dump([seed_types, locations], f)
                count += 1

            cons2 = ({'type': 'ineq', "fun": cons })

            res = optimize.minimize(func, x0=x, method="SLSQP", bounds=bounds, constraints=cons2, callback=cbf, options={'maxiter': 100})

            locations = res.x.reshape([-1,2])
            points = res.x.reshape([-1, 2])
            labels = seed_types

            with open("data/f_seed"+str(NUM_SEEDS)+"_overlap"+str(OVERLAP_PERCENTAGE)+"_trial"+str(initial_try_id), "wb") as f:
                pickle.dump([seed_types, locations], f)
