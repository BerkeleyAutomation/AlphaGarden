import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import argparse
import imutils
import math
import glob
from movement import batch_target_approach
from control import start, MyHandler, mount_xPruner, mount_yPruner, dismount_xPruner, dismount_yPruner, mount_nozzle, dismount_nozzle, photo
from thread import FarmBotThread
import pickle
import argparse
from urllib.request import urlopen
from datetime import datetime, timezone
import os
from time import gmtime, strftime
import time
from PIL import Image, ImageOps
import sys

GARDEN_START_DATE = 1625526000

def auto_irrigate_withsim():
    #convert sectors to farmbot coordinates
    #call movement functions

    utc_time = datetime.now(timezone.utc)
    utc_timestamp = utc_time.timestamp()
    most_recent = round(utc_timestamp)

    days = (most_recent - GARDEN_START_DATE) // (1800 * 24)
    os.system('python3 ../Learning/eval_policy.py -p i -s 1 -d 50')

    timestep = days

    with open('./policy_metrics/auto_irrigate/watered_sectors' + '_' + str(timestep) + '.pkl','rb') as f:
        sectors = pickle.load(f)
        print(len(sectors), sectors)

    fb = FarmBotThread()
    print(sectors)

    for i in sectors:
        #order x coordinates
        print((int(i[0]) * 10, int(i[1]) * 10,0))
        farmbotx = int(i[0] * 10 * (274.66/2/150))
        farmboty = int(i[1] * 10 * (125.28/150))

        fb.update_action("move", (farmbotx, farmboty,0)) #sim to farmbot coord * scaling factor
        time.sleep(100)
        fb.update_action("water", None) #sim to farmbot coord * scaling factor
        #farmbot watering action
    
    return

def watergrid_oneday_lookahead(sim2FB, timestep=0):
    #load garden state
    #os.system('python3 ../Learning/create_state.py -t ' + str(timestep))
    os.system('python3 ../Learning/create_state.py')
    time.sleep(2)
    os.system('python3 ../Learning/eval_policy.py -p ba -s 1 -d 1')
    timestep = pickle.load(open("/Users/mpresten/Desktop/AlphaGarden_git/AlphaGarden/Center-Tracking/timestep.p", "rb")) #change path accordingly

    with open('./policy_metrics/auto_irrigate/watered_sectors' + '_' + str(timestep) + '.pkl','rb') as f:
        sectors = pickle.load(f)
        print(len(sectors), sectors)

    fb = FarmBotThread()

    for i in sectors:
        print((int(i[0]) * 10, int(i[1]) * 10,0))
        farmbotx = sim2FB[i[0] + i[1]][0]
        farmboty = sim2FB[i[0] + i[1]][1]

        fb.update_action("move", (farmbotx, farmboty,0)) #sim to farmbot coord * scaling factor
        time.sleep(60) 
        fb.update_action("water", None) 

    fb.update_action("move", (0,0,0))

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--timestep', type=int, default=0)
    # args = parser.parse_args()
    #watergrid_oneday_lookahead(args.timestep)
    f = sys.argv[1]
    if f == 'r':
        sim2FB = {45.0: (1245, 993), 40.0: (1153, 1119), 108.0: (1153, 551), 82.0: (1126, 793), 150.0: (1062, 283), 91.0: (860, 960), 185.0: (778, 250), 142.0: (604, 768), 114.0: (549, 1052), 225.0: (549, 125), 191.0: (512, 442), 134.0: (311, 1102), 194.0: (265, 643), 242.0: (265, 242), 156.0: (146, 1069), 173.0: (119, 952)}
    elif f == 'l':
        sim2FB = {45.0: (1501, 993), 40.0: (1593, 1119), 108.0: (1593, 551), 82.0: (1620, 793), 150.0: (1684, 283), 91.0: (1885, 960), 185.0: (1968, 250), 142.0: (2142, 768), 114.0: (2197, 1052), 225.0: (2197, 125), 191.0: (2233, 442), 134.0: (2435, 1102), 194.0: (2481, 643), 242.0: (2481, 242), 156.0: (2600, 1069), 173.0: (2627, 952)}
    else:
        print("ERROR, specify 'r' or 'l'")

    watergrid_oneday_lookahead(sim2FB)


