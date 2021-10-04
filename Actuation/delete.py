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
import pickle as pkl

def depth_sensor(vol):
    # 688 - 10 cm
    # 368 - 20 cm
    # 285 - 30 cm
    # 240 - 40 cm
    # 199 - 50 cm
    x = np.array([688, 368, 285, 240, 199])
    y = np.array([10, 20, 30, 40, 50])
    z = np.polyfit(x,y,3)
    p = np.poly1d(z)
    return p(vol)

def perpendiculars(target_list):
    angles = []
    for i in range(len(target_list)):
        center, target = target_list[i][0], target_list[i][1]
        k = np.array([-1 * (target[0] - center[0]), target[1] - center[1]])
        print(k)
        k /= np.linalg.norm(k)
        print(k)
        x = np.random.randn(2)  # take a random vector
        x -= x.dot(k) * k       # make it orthogonal
        x /= np.linalg.norm(x)
        print(x)
        dot_product = np.dot(np.array([-1, 0]), x) #[-1, 0] is the -x axis which is 0 degree
        if x[1] >= 0:
            angle = np.arccos(dot_product) * (180/math.pi)
        else:
            angle = 360 - (np.arccos(dot_product) * (180/math.pi))
        angle = angle if angle < 180 else angle - 180
        angles.append(angle)
        print(angle, "\n")
    print(angles)
    return angles

if __name__ == '__main__':
    # new_x = np.array([725, 465, 372, 358])
    # new_y = np.array([])
    # z = np.polyfit(old_x,old_y,3)
    # p = np.poly1d(z)
    # print(p(311))
    # xp = np.linspace(800, 100, 100)
    # _ = plt.plot(xp, p(xp))
    # plt.show()
    fb = FarmBotThread()
    fb.update_action("water", None) #prune with angle

    # fb.update_action('read_pin', 8)
    # time.sleep(3)
    # depth = pkl.load(open('./data/read_water.p', 'rb'))
    # print(depth)

    # fb.update_action("prune_scissor", None) #prune with angle
    # time.sleep(11)
    # print('done')
    # os.system('python3 ../Learning/create_state.py ' + 'r')
    # os.system('python3 ../Learning/eval_policy.py -p ba -d 2')

    ### SCISSOR OFFSET AND ANGLE TESTING
    # pos_x, pos_y = 110, 47
    # ang_sf = (pos_x-pos_y)/90
    # sci_rad = 12
    # #for thsi target, 3 cm wrong in x (should be 11)
    # # target_list = [[(2962.110887096774, 881.8161290322575), (2958.604838709677, 1127.2395161290317)]]
    # target_list = [[(2955.0987903225805, 864.2858870967736), (3056.774193548387, 1081.6608870967736)]]
    # actual_farmbot_coords = [(30,30)]
    # angles = perpendiculars(target_list)
    # offset = [-1 *sci_rad*math.sin(angle*math.pi/180) - 1 for angle in angles]

    # for cur_point, angle in zip(actual_farmbot_coords, angles):

    #     # offset = [-1 *sci_rad*math.sin(angle*math.pi/180) - 1 for angle in angles]
    #     print("ANGLE: ", angle)
    #     x, y = 300, 300
    #     fb = FarmBotThread()
    #     fb.update_action("move", (x, y, 0))
    #     time.sleep(5)

    #     mod_angle = (angle - 90) * ang_sf + pos_y if angle > 90 else pos_y - ang_sf * (90 - angle)
    #     # mod_angle = 0
    #     print("---CURR PT, MOD_ANGLE: ", mod_angle)

    #     fb.update_action("servo", (11, mod_angle)) #move scissors to corrected angle according to real-life contraints
    #     time.sleep(2)

    #     scissors_offset = (sci_rad*math.cos(angle*math.pi/180) + 2, -1 *sci_rad*math.sin(angle*math.pi/180) - 1) #scissor offset
    #     print("---Scissor offset: ", (scissors_offset[0] * 10, scissors_offset[1]*10,0))
    #     fb.update_action("move_rel", (scissors_offset[0] * 10, scissors_offset[1]*10,0)) #perform 
    #     time.sleep(5)

