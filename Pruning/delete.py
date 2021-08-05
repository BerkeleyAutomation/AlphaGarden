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
    return (-0.1446163673788062 * vol) + 77.7510824047129

if __name__ == '__main__':
    # x = np.array([688, 368, 285, 240, 199])
    # y = np.array([10, 20, 30, 40, 50])
    # z = np.polyfit(x,y,3)
    # p = np.poly1d(z)
    # print(p(303))
    fb = FarmBotThread()
    # fb.update_action('read_pin', 54)
    # time.sleep(3)
    # depth = pkl.load(open('./data/read_depth.p', 'rb'))
    # print(depth)
    fb.update_action("prune_scissor", None) #prune with angle
    time.sleep(11)
    print('done')
