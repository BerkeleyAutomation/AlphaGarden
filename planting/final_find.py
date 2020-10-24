import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pickle
from scipy import optimize
from seed_solver import find_labels
import os
from plant_presets import *

# Parameters here
GARDEN_SIZE = 150
alphas = [0.2, 0.25, 0.1, 0.05, 0.025, 0.0]
num_trials = 3

RATIO = 5
CONSTANT = 1.0
np.random.seed(10)

# The final placement result:
initial_try_id = 1
NUM_SEEDS = 60


# for NUM_SEEDS in [60]:
labels = np.asarray([i%len(PLANTS) for i in range(NUM_SEEDS)])
for initial_try_id in range(num_trials):
    if not os.path.exists("data/"+str(NUM_SEEDS)+"data"):
        if not os.path.exists("data/"+str(NUM_SEEDS)+"data-"+str(initial_try_id)):
            np.random.shuffle(labels)
            [points, labels, score] = find_labels(NUM_SEEDS=NUM_SEEDS, labels=labels)
            with open("data/"+str(NUM_SEEDS)+"data-"+str(initial_try_id), "wb") as f:
                pickle.dump([labels, points], f)
        with open("data/"+str(NUM_SEEDS)+"data-"+str(initial_try_id), "rb") as f:
            [labels, points] = pickle.load(f)

        # [aa, bb] = np.random.choice(len(labels), 2)
        for OVERLAP_PERCENTAGE in alphas:
            seed_types = np.asarray(labels)
            seed_locs = np.asarray(points)
            max_radius = np.asarray([PLANT_SIZE[PLANTS[i]] for i in seed_types])
            tmp = max_radius.reshape([-1,1]).repeat(NUM_SEEDS, 1)
            r_max_sum = (tmp + tmp.transpose())

            R = np.zeros([NUM_SEEDS, NUM_SEEDS])
            for i, v1 in enumerate(seed_types):
                for j, v2 in enumerate(seed_types):
                    R[i,j] = PLANTS_RELATION[PLANTS[v1]][PLANTS[v2]]

            garden = Image.new("RGB", (100*RATIO,100*RATIO), (255,255,255))
            dr = ImageDraw.Draw(garden)
            for i in range(len(seed_locs)):
                [x,y] = seed_locs[i]
                l = seed_types[i]
                type = PLANTS[l]
                r = PLANT_SIZE[type]
                color = COLORS[l]
                dr.ellipse(((x-r)*RATIO,(y-r)*RATIO,(x+r)*RATIO,(y+r)*RATIO), color)

            seed = Image.new("RGB", (100*RATIO,100*RATIO), (255,255,255))
            dr = ImageDraw.Draw(seed)
            for i in range(len(seed_locs)):
                [x,y] = seed_locs[i]
                l = seed_types[i]
                type = PLANTS[l]
                r = PLANT_SIZE[type]
                color = COLORS[l]
                dr.ellipse((x*RATIO-3,y*RATIO-3,x*RATIO+3,y*RATIO+3), color)
                dr.ellipse(((x-r)*RATIO,(y-r)*RATIO,(x+r)*RATIO,(y+r)*RATIO), fill=None, outline=color, width=1)

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
                if count % 20 == 0:
                    print(NUM_SEEDS, OVERLAP_PERCENTAGE, count, func(Xi))
                count += 1

            cons2 = ({'type': 'ineq', "fun": cons })

            res = optimize.minimize(func, x0=x, method="SLSQP", bounds=bounds, constraints=cons2, callback=cbf, options={'maxiter': 100})

            locations = res.x.reshape([-1,2])
            points = res.x.reshape([-1, 2])
            labels = seed_types

            with open("data/seed"+str(NUM_SEEDS)+"_overlap"+str(OVERLAP_PERCENTAGE)+"_trial"+str(initial_try_id), "wb") as f:
                pickle.dump([seed_types, locations], f)
