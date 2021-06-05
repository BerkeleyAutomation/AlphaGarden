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
import argparse
import time

def separate_list(target_list):
    x_list, y_list = [], []
    for i in target_list:
        target, center = i[0], i[1]
        if np.abs(target[0] - center[0]) > np.abs(target[1] - center[1]):
            y_list.append(target)
        else:
            x_list.append(target)

    return x_list, y_list


def batch_prune(target_list, overhead, rpi_check):
    fb = FarmBotThread()
    # actual_farmbot_coords = batch_target_approach(fb, target_list, overhead)
    # print("ACTUAL FARMBOT COORDS: ", actual_farmbot_coords)
    # x_list, y_list = separate_list(target_list)
    # print("x_list: ", x_list)
    # print("y_list: ", y_list)

    x_list = [(96,73), (60, 45), (73,65)]
    y_list = [(74,65),(83,101), (69,97), (94,98), (18,58)]

    # dismount_nozzle()
    # mount_yPruner()
    response = input("Enter 'y' after MOUNTING yPruner.")

    for i in y_list:
        response = input()
        fb.update_action("move", (i[0] * 10 - 40, i[1] * 10 + 70,0))    #y requires offset
        response = input("Enter 'y' when READY to prune.")
        if False: #rpi_check:
            done = False
            inc = -250
            while (done == False):
                #go down z cm prune and come back up
                bef_name = recent_rpi_photo(fb)
        
                fb.update_action("prune", None)
                fb.update_action("move_rel", (0,0,inc))
                fb.update_action("move_rel", (0,0,(-1 * inc) - 1))
                fb.update_action("prune", None)

                aft_name = recent_rpi_photo(fb)
                chk = check_prune(bef_name, aft_name)
                print("--PRUNE CHECK--: ", chk)

                response = input("Enter 'y' if leaf pruned, else 'n': ")
                if response == 'y':
                    done = True
                inc -= 50

        else:
            #TODO add functionality to go up and down and prune
            fb.update_action("prune", None)
            fb.update_action("move_rel", (0,0,-390))
            fb.update_action("move_rel", (0,0, 389))
            fb.update_action("prune", None)

    # dismount_yPruner()
    # mount_xPruner()
    response = input("Enter 'y' after pruner is SWITCHED.")

    for i in x_list:
        fb.update_action("move", (i[0] * 10, i[1] * 10,0))
        response = input("Enter 'y' when READY to prune.")
        if False: # rpi_check: (can't use servo'ing with xPruner bc it blocks camera)
            done = False
            inc = -250
            while (done == False):
                #go down z cm prune and come back up
                bef_name = recent_rpi_photo(fb)
        
                fb.update_action("prune", None)
                fb.update_action("move_rel", (0,0,inc))
                fb.update_action("move_rel", (0,0,(-1 * inc) - 1))
                fb.update_action("prune", None)

                aft_name = recent_rpi_photo(fb)
                chk = check_prune(bef_name, aft_name)
                print("PRUNE CHECK: ", chk)

                response = input("Enter 'y' if leaf pruned, else 'n': ")
                if chk and response == 'y':
                    done = True
                inc -= 50
        else:
            #TODO add functionality to go up and down and prune
            fb.update_action("prune", None)
            fb.update_action("move_rel", (0,0,-390))
            fb.update_action("move_rel", (0,0, 389))
            fb.update_action("prune", None)

    dismount_xPruner()
    mount_nozzle()

    return None

def crop_o_px_to_cm(x_px, y_px):
    pred_pt = (round(274.66 - (x_px - 102)/11.9), round((y_px - 72)/11.9))
    return pred_pt

def recent_rpi_photo(fb):
    fb.update_action("photo", None)
    cwd = os.getcwd()
    rpi_folder_path = os.path.join(cwd, "rpi_images")
    time.sleep(15)
    photo(rpi_folder_path + "/")
    time.sleep(5)
    list_of_files = glob.glob(rpi_folder_path + '/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file[latest_file.find("rpi_images")+11:]

def check_prune(bef_rpi, aft_rpi):
    #checks whether leaf has been pruned


    cwd = os.getcwd()
    image_path  = os.path.join(cwd, "rpi_images", bef_rpi + "_resized.jpg")
    bef_rpi = cv2.imread(image_path, 1)

    image_path  = os.path.join(cwd, "rpi_images", aft_rpi + "_resized.jpg")
    aft_rpi = cv2.imread(image_path, 1)

    meth = 'cv2.TM_CCOEFF_NORMED'
    threshold = 0.8 #threshold for normalized ccoeff if pruned or not

    bef_rpi = bef_rpi.copy()
    aft_rpi = aft_rpi.copy()

    method = eval(meth)

    # Apply template Matching

    res = cv2.matchTemplate(bef_rpi, aft_rpi.astype(np.uint8), method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print("MAX_VAL: ", max_val)
    prune = True if max_val < threshold else False
    return prune


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--overhead", "-o", type=str, default="", help="Specify the overhead")
    parser.add_argument("--rpi_check_prune", "-p", type=bool, default=False, help="Use rpi images to check if the tool correctly pruned the target leaf")

    args = parser.parse_args()

    #print(get_points(im)) #Find Points to crop in overhead image
    # target_list = [((2481.782258064516, 923.8887096774195), (2000, 900)), ((2250.383064516129, 678.4653225806451), (2100, 600)), ((3098.8467741935483, 1337.6024193548387), (3000, 1000))]
    target_list = [((2192.1348713372863, 1033.2203931840422), (2051.2697426745726, 766.4407863680847)), ((2646.372097874638, 677.2447216554123), (2762.744195749276, 475.4894433108245)), ((2469.969855260587, 922.4155430868436), (2405.9397105211738, 680.8310861736873)), ((2469.4256064016004, 993.9150318829708), (2579.8512128032007, 965.8300637659415)), ((2358.408583699892, 1373.3575856917587), (2478.817167399784, 1297.7151713835174)), ((2533.076888573078, 1331.9534684904738), (2850.1537771461562, 1248.9069369809476)), ((2211.2047908751406, 1345.0958580073393), (2084.409581750281, 1262.1917160146788)), ((3174.827852473074, 860.0234503457663), (2972.6557049461485, 877.0469006915325))]

    batch_prune(target_list, args.overhead, args.rpi_check_prune)
    
