import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import argparse
import imutils
import math
import glob
from movement import batch_target_approach, correct_image, get_points
from control import start, MyHandler, mount_xPruner, mount_yPruner, dismount_xPruner, dismount_yPruner, mount_nozzle, dismount_nozzle, photo
from thread import FarmBotThread
import argparse
import time
import pickle as pkl

def separate_list(actual_coords, target_list):
    x_list, y_list = [], []
    for i in range(len(target_list)):
        center, target = target_list[i][0], target_list[i][1]
        if np.abs(target[0] - center[0]) > np.abs(target[1] - center[1]): #if x_diff > y_diff
            y_list.append(actual_coords[i])
        else:
            x_list.append(actual_coords[i])

    return x_list, y_list

def batch_prune(target_list, overhead, rpi_check):
    fb = FarmBotThread()
    # actual_farmbot_coords = batch_target_approach(fb, target_list, overhead)
    # print("--ACTUAL FARMBOT COORDS: ", actual_farmbot_coords)
    # x_list, y_list = separate_list(actual_farmbot_coords, target_list)
    # print("--x_list: ", x_list)
    # print("--y_list: ", y_list)

    x_list = [(15, 37), (78, 63), (51, 15)]
    y_list = [(50, 66.9), (88, 88)] #(93, 50.9)

    # dismount_nozzle()
    # mount_yPruner()
    response = input("===== Enter 'y' after MOUNTING yPruner.")

    for i in y_list:
        fb.update_action("move", (i[0] * 10 - 40, i[1] * 10 + 70,0))    #y requires offset
        response = input("===== Enter 'y' when READY to prune.")
        if rpi_check:
            done = False
            inc = -350
            while (done == False):
                #go down z cm prune and come back up
                bef_name = recent_rpi_photo(fb)
        
                # fb.update_action("prune", None)
                # fb.update_action("move_rel", (0,0,inc))
                # fb.update_action("move_rel", (0,0,(-1 * inc) - 1))
                # fb.update_action("prune", None)
                fb.update_action("prune", None)
                fb.update_action("move_rel", (0,0,-250))
                fb.update_action("move_rel", (0,0, 249))
                fb.update_action("prune", None)
                time.sleep(180) #modify!!!
                print("PHOTO TIME")

                aft_name = recent_rpi_photo(fb)
                chk = check_prune(i, bef_name, aft_name)
                print("--PRUNE CHECK--: ", chk)

                if chk:
                    done = True
                done = True
                inc -= 50

        else:
            #TODO add functionality to go up and down and prune
            fb.update_action("prune", None)
            fb.update_action("move_rel", (0,0,-390))
            fb.update_action("move_rel", (0,0, 389))
            fb.update_action("prune", None)
            time.sleep(180)

    # dismount_yPruner()
    # mount_xPruner()
    response = input("===== Enter 'y' after pruner is SWITCHED.")

    for i in x_list:
        fb.update_action("move", (i[0] * 10, i[1] * 10,0))
        response = input("===== Enter 'y' when READY to prune.")
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
                time.sleep(180)

                aft_name = recent_rpi_photo(fb)
                chk = check_prune(i, bef_name, aft_name)
                print("PRUNE CHECK: ", chk)

                response = input("===== Enter 'y' if leaf pruned, else 'n': ")
                if chk and response == 'y':
                    done = True
                inc -= 50
        else:
            #TODO add functionality to go up and down and prune
            fb.update_action("prune", None)
            fb.update_action("move_rel", (0,0,-250))
            fb.update_action("move_rel", (0,0, 249))
            fb.update_action("prune", None)
            time.sleep(180) #modify!!!

    # dismount_xPruner()
    # mount_nozzle()

    return None

def perpendiculars(target_list):
    angles = []
    for i in range(len(target_list)):
        center, target = target_list[i][0], target_list[i][1]
        k = np.array([-1 * (target[0] - center[0]), target[1] - center[1]])
        print(k)
        print(np.linalg.norm(k))
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

def batch_prune_scissors(target_list, overhead, rpi_check):
    pos_x, pos_y = 110, 47 #47
    ang_sf = (pos_x-pos_y)/90
    sci_rad = 13
    angles = perpendiculars(target_list)
    offset = [-1 *sci_rad*math.sin(angle*math.pi/180) - 1 for angle in angles]

    fb = FarmBotThread()
    # Reset to good position
    fb.update_action("servo", (6, 101))
    fb.update_action("servo", (11, 0))

    # Start Locationing
    actual_farmbot_coords = batch_target_approach(fb, target_list, overhead, offset)
    # actual_farmbot_coords = [(118, 73), ()] 
    print("--ACTUAL FARMBOT COORDS: ", actual_farmbot_coords)
    height_fb_clearance = 8 #cm from top of farmbot

    for cur_point, angle in zip(actual_farmbot_coords, angles):
        # Reset to good position
        fb.update_action("servo", (6, 101))
        fb.update_action("servo", (11, 0))
        mod_angle = (angle - 90) * ang_sf + pos_y if angle > 90 else pos_y - ang_sf * (90 - angle)
        print("---CURR PT, MOD_ANGLE: ", cur_point, mod_angle)

        fb.update_action("servo", (11, mod_angle)) #move scissors to corrected angle according to real-life contraints
        time.sleep(2)

        print("---TIME TO CALC DEPTH")
        dsensor_adjusted = tuple((cur_point[0] - 1.5, cur_point[1])) #depth sensor offset
        fb.update_action("move", (dsensor_adjusted[0] * 10, dsensor_adjusted[1] * 10,0))
        time.sleep(40)
        print("---DONE SLEEPING")
        z = get_depth(fb)
        time.sleep(2)
        print("---Depth: ", z)
        z = min(z, 40)

        fb.update_action("move_rel", (15, 0,0)) #reset to account for depth sensor
        time.sleep(10)

        scissors_offset = (sci_rad*math.cos(angle*math.pi/180) + 2, -1 *sci_rad*math.sin(angle*math.pi/180) - 1) #scissor offset
        print("---Scissor offset: ", (scissors_offset[0] * 10, scissors_offset[1]*10,0))
        fb.update_action("move_rel", (scissors_offset[0] * 10, scissors_offset[1]*10,0)) #perform scissors offset
        
        prune_top = False
        if z < height_fb_clearance:
            prune_top = True
            print("---PRUNING TOP")
            response = input("===== Enter 'n' if you don't want to prune top.")
            if response == 'n':
                prune_top = False

        response = input("===== Enter 'y' when READY to prune.")

        if prune_top: #FIX - add orientation
            fb.update_action("move_rel", (0, -100, 0))#move to z position from the depth sensor after setting up the scissors
            time.sleep(15)
            fb.update_action("move_rel", (0, 0, (z * -10)))#move to z position from the depth sensor after setting up the scissors
            time.sleep(20)
            fb.update_action("move_rel", (0, 100, 0))#move to z position from the depth sensor after setting up the scissors
            time.sleep(15)
        else:
            fb.update_action("servo", (6, 38)) # Ordinary Scissor cut
            time.sleep(2)
            fb.update_action("move_rel", (0, 0, (z * -10)+ 70))#move to z position from the depth sensor after setting up the scissors
            time.sleep(90)

        print("---TIME TO CUT")
        if prune_top:
            # while (done == False):
            fb.update_action("prune_scissor", None) #prune with angle
            time.sleep(11)
                # done = prune_check_sensor(fb, z, dsensor_adjusted, scissors_offset)
        else:
            time.sleep(2)
            # while (done == False):
            fb.update_action("prune_scissor", None) #prune with angle
            time.sleep(11)
            fb.update_action("move_rel", (0, 0, (z * 10) - 70.5))#move to z position from the depth sensor after setting up the scissors
            time.sleep(90)
                # done = prune_check_sensor(fb, z, dsensor_adjusted, scissors_offset)
        print("--COMPLETE")
    return None

def prune_check_sensor(fb, prev_depth, origi_pos, scissors_offset):
    #use origi_pos to check the depth with consistent offset
    fb.update_action("move", (origi_pos[0] * 10, origi_pos[1] * 10,0))
    time.sleep(15)
    print(origi_pos)
    fraction = .7
    curr_depth = get_depth(fb)
    time.sleep(2)
    print(curr_depth)
    if curr_depth * fraction < prev_depth:
        return True
    fb.update_action("move_rel", (scissors_offset[0] *10, scissors_offset[1]*10,0))
    time.sleep(5)
    return False

def get_depth(fb):
    #get depth necessary to prune the leaf with the depth sensor
    #depth offset
    fb.update_action('read_pin', 54)
    time.sleep(3)
    value = pkl.load(open('./data/read_depth.p', 'rb'))
    return value

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

def check_prune(i, bef_rpi, aft_rpi):
    #checks whether leaf has been pruned
    cwd = os.getcwd()
    # image_path  = os.path.join(cwd, "rpi_images", bef_rpi + "_resized.jpg")
    image_path  = os.path.join(cwd, "rpi_images", bef_rpi)
    bef_rpi = cv2.imread(image_path, 1)

    # image_path  = os.path.join(cwd, "rpi_images", aft_rpi + "_resized.jpg")
    image_path  = os.path.join(cwd, "rpi_images", aft_rpi)
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
    file_name = "threshold.txt"
    f = open(cwd + '/' + file_name, "r")
    item = f.read()
    fil = open(cwd + '/' + file_name, "w+")
    item = str(item) + "[" + str(i) + ", " + str(max_val) + "]"
    fil.write(item)
    fil.close()

    prune = True if max_val < threshold else False
    return prune

def potted_plant_manual(overhead):
    im = cv2.cvtColor(cv2.imread(overhead), cv2.COLOR_BGR2RGB)
    im = correct_image(im, (350.74890171959316, 596.1321074432035), (3998.9477218526417, 609.436990084097), (4006.9306514371774, 2371.0034517384215), (318.81718338144833, 2325.7668507593826))
    center, target = get_points(im)
    print("--Center: ", center)
    print("--Target: ", target)
    batch_prune_scissors([(center, target)], overhead, False)

def potted_plant_auto(overhead, mask):
    file = overhead[-22:-4]
    im = cv2.cvtColor(cv2.imread(overhead), cv2.COLOR_BGR2RGB)
    im = correct_image(im, (350.74890171959316, 596.1321074432035), (3998.9477218526417, 609.436990084097), (4006.9306514371774, 2371.0034517384215), (318.81718338144833, 2325.7668507593826))
    plt.imsave(file + "_cropped.jpg", im)
    ## Identify all key points (need prior from get_points, and manual mask)
    print("INSTRUCTION: Label center then outer most point!")
    center, outer = get_points(im)
    dist = ((center[0] - outer[0])**2 + (center[1] - outer[1])**2)**0.5
    prior = {'external': [{'circle': (center, dist, target), 'days_post_germ': 40}]}
    pkl.dump(prior, open('./Experiments/prior' + file + '.p', 'wb'))
    leaf_centers = get_keypoints(mask, overhead, './Experiments/prior' + file + '.p', "../Center-Tracking/models/leaf_keypoints.pth")
    generate_image(leaf_centers, overhead)
    ## Cut all points
    return

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--overhead", "-o", type=str, default="", help="Specify the overhead")
    parser.add_argument("--rpi_check_prune", "-p", type=bool, default=False, help="Use rpi images to check if the tool correctly pruned the target leaf")
    parser.add_argument("--mask", "-m", type=str, default="", help="Pass in mask for potted plant experiments.")

    args = parser.parse_args()

    # Initialize - comment out after failure
    # pkl.dump([], open("actual_coords.p", "wb"))

    #target_l = pkl.load(open("/Users/mpresten/Desktop/AlphaGarden_git/AlphaGarden/Center-Tracking/current_pts.p", "rb"))
    #print(target_l)
    target_list = [((1947.0, 994.0), (1926.0, 1143.0)), ((2988.0, 1427.0), (2924.0, 1368.0))]

    batch_prune_scissors(target_list, args.overhead, args.rpi_check_prune)

    ### External Pot
    # potted_plant_manual(args.overhead)
    
