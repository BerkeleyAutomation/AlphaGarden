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
    actual_farmbot_coords = batch_target_approach(fb, target_list, overhead)
    print("--ACTUAL FARMBOT COORDS: ", actual_farmbot_coords)
    x_list, y_list = separate_list(actual_farmbot_coords, target_list)
    # print("--x_list: ", x_list)
    # print("--y_list: ", y_list)

    # x_list = [(15, 37), (78, 63), (51, 15)]
    # y_list = [(50, 66.9), (88, 88)] #(93, 50.9)

    # dismount_nozzle()
    # mount_yPruner()
    for i in y_list:
        response = input("===== Enter 'y' in yPruner MOUNTED.")

        fb.update_action("move", (i[0] * 10 - 40, i[1] * 10 + 70,0))    #y requires offset
        response = input("===== Enter 'y' when READY to prune.")
        if False:
            done = False
            inc = -350
            while (done == False):
                #go down z cm prune and come back up
                bef_name = recent_rpi_photo(fb)
        
                fb.update_action("prune", None)
                fb.update_action("move_rel", (0,0,-250))
                fb.update_action("move_rel", (0,0, 249))
                fb.update_action("prune", None)
                time.sleep(180) #modify!!!
                print("PHOTO TIME")

                aft_name = recent_rpi_photo(fb)
                chk = compare_recent_rpi(i, bef_name, aft_name)
                print("--PRUNE CHECK--: ", chk)

                if chk:
                    done = True
                done = True
                inc -= 50
        else:
            print("---TIME TO CALC DEPTH")
            dsensor_adjusted = tuple((i[0] - 1.5, i[1])) #depth sensor offset
            fb.update_action("move", (dsensor_adjusted[0] * 10, dsensor_adjusted[1] * 10,0))
            time.sleep(5)
            print("---DONE SLEEPING")
            z = get_depth(fb)
            time.sleep(3)
            print("---Depth: ", z)
            z = min(z, 40)
            fb.update_action("prune", None)
            fb.update_action("move_rel", (0,0,(z * -10)))
            fb.update_action("move_rel", (0,0,(z * 10)))
            fb.update_action("prune", None)
            time.sleep(180)

    # dismount_yPruner()
    # mount_xPruner()

    for i in x_list:
        response = input("===== Enter 'y' in xPruner MOUNTED.")
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
                chk = compare_recent_rpi(i, bef_name, aft_name)
                print("PRUNE CHECK: ", chk)

                response = input("===== Enter 'y' if leaf pruned, else 'n': ")
                if chk and response == 'y':
                    done = True
                inc -= 50
        else:
            print("---TIME TO CALC DEPTH")
            dsensor_adjusted = tuple((i[0] - 1.5, i[1])) #depth sensor offset
            fb.update_action("move", (dsensor_adjusted[0] * 10, dsensor_adjusted[1] * 10,0))
            time.sleep(5)
            print("---DONE SLEEPING")
            z = get_depth(fb)
            time.sleep(3)
            print("---Depth: ", z)
            z = min(z, 40)
            fb.update_action("prune", None)
            fb.update_action("move_rel", (0,0,(z * -10)))
            fb.update_action("move_rel", (0,0,(z * 10)))
            fb.update_action("prune", None)
            time.sleep(180)
    # dismount_xPruner()
    # mount_nozzle()

    return None

def perpendiculars(target_list):
    angles, k_arr = [], []
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
        k_arr.append(k)
        print(angle, "\n")
    print(angles)
    return angles, k_arr

def batch_prune_scissors(target_list, overhead, rpi_check):
    pos_x, pos_y = 110, 47 #47
    ang_sf = (pos_x-pos_y)/90
    sci_rad = 13
    angles, k_arr = perpendiculars(target_list) #angles=angle of cut, k_arr=vector from center to target pt for repositioning
    offset = [-1 *sci_rad*math.sin(angle*math.pi/180) - 1 for angle in angles]

    fb = FarmBotThread()
    # Reset to good position
    fb.update_action("servo", (6, 101))
    fb.update_action("servo", (11, 0))

    # Start Locationing
    # actual_farmbot_coords = batch_target_approach(fb, target_list, overhead, offset)
    actual_farmbot_coords = [(246, 58), (198, 6), (234, 1), (154, 94), (187, 43)]
    print("--ACTUAL FARMBOT COORDS: ", actual_farmbot_coords)
    height_fb_clearance = 10 #cm from top of farmbot

    for cur_point, angle, k in zip(actual_farmbot_coords, angles, k_arr):
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
        time.sleep(50)
        print("---DONE SLEEPING")
        z = get_depth(fb)
        time.sleep(2)
        print("---Depth: ", z)
        z = min(z, 35)

        fb.update_action("move_rel", (15, 0,0)) #reset to account for depth sensor

        time.sleep(10)
        curr_rpi = recent_rpi_photo(fb) #name of rpi image of current state
        
        scissors_offset = (sci_rad*math.cos(angle*math.pi/180) + 2, -1 *sci_rad*math.sin(angle*math.pi/180) - 1) #scissor offset
        print("---Scissor offset: ", (scissors_offset[0] * 10, scissors_offset[1]*10,0))

        prune_top = False
        if z < height_fb_clearance:
            prune_top = True
            print("---PRUNING TOP")
            response = input("===== Enter 'n' if you don't want to prune top.")
            if response == 'n':
                prune_top = False
                fb.update_action("move_rel", (scissors_offset[0] * 10, scissors_offset[1]*10,0)) #perform scissors offset
        else:
            scissors_offset = (sci_rad*math.cos(angle*math.pi/180) + 2, -1 *sci_rad*math.sin(angle*math.pi/180) - 1) #scissor offset
            fb.update_action("move_rel", (scissors_offset[0] * 10, scissors_offset[1]*10,0)) #perform scissors offset

        response = input("===== Enter 'y' when READY to prune.")

        if prune_top: 
            fb.update_action("servo", (6, 101)) # Ordinary Scissor cut
            time.sleep(1)
            fb.update_action("servo", (11, 70))
            time.sleep(1)
            fb.update_action("move_rel", (200, -180, 0))
            time.sleep(25)
            fb.update_action("move_rel", (0, 0, (z * -10)+30))#move to z position from the depth sensor after setting up the scissors
            time.sleep(30)
            fb.update_action("move_rel", (-200, 180, 0))
            time.sleep(25)
        else:
            fb.update_action("servo", (6, 38)) # Ordinary Scissor cut
            time.sleep(2)
            fb.update_action("move_rel", (0, 0, (z * -10)+ 30))#move to z position from the depth sensor after setting up the scissors
            time.sleep(90)

        print("---TIME TO CUT")
        done = False
        i = 0 #counter for number of times repositioned scissors for same cut
        while (done == False and i < 2):  #change iteration threshold
            if prune_top:
                fb.update_action("prune_scissor", None)
                time.sleep(11)
                fb.update_action("move_rel", (0, 0, (z * 10)- 30.5))#move to z position from the depth sensor after setting up the scissors
                time.sleep(20)
                done = True
            else:
                # fb.update_action("servo", (6, 38)) # Ordinary Scissor cut
                # time.sleep(2)
                fb.update_action("prune_scissor", None) #prune with angle
                time.sleep(11)
                fb.update_action("move_rel", (0, 0, (z * 10) - 30.5))#move to z position from the depth sensor after setting up the scissors
                time.sleep(90)
                done = True
            # done, curr_rpi = prune_check_sensor(fb, i, curr_rpi, z, cur_point, dsensor_adjusted, scissors_offset)
            # print("---DONE: ", done)
            # retry = input("==== enter y to continue")
        #     if True: #retry == 'y':
        #         continue
        #     z = reposition_scissors(fb, k, scissors_offset, cur_point)
        #     i += 1
        # print("--COMPLETE")
    return None

def reposition_scissors(fb, k, scissors_offset, rpi_pos):
    #reposition scissors to move in direction of the center
    reposition_dist = 2 #length of reposition vector in cm
    change = [k[0]*np.sqrt(reposition_dist), k[1]*np.sqrt(reposition_dist)]
    fb.update_action("move_rel", (scissors_offset[0] *10, scissors_offset[1]*10,0))
    time.sleep(5)
    z = get_depth(fb)
    z = min(z, 40)
    fb.update_action("move_rel", (change[0] *10, change[1]*10,(z * -10)+ 70))
    
    time.sleep(5)
    return z

def prune_check_sensor(fb, i, prev_rpi, prev_depth, rpi_pos, depthsen_pos, scissors_offset):
    #use depthsen_pos to check the depth with consistent offset
    fb.update_action("move", (depthsen_pos[0] * 10, depthsen_pos[1] * 10,0)) #move to depth sensor
    time.sleep(15)
    print(depthsen_pos)
    epsilon = 3 # min depth difference threshold in cm
    curr_depth = get_depth(fb)
    time.sleep(2)
    print(curr_depth)
    if curr_depth - prev_depth > epsilon: #curr_depth has to be lower if leaf was cut
        return True
    fb.update_action("move_rel", (15, 0,0)) #reset to account for depth sensor

    curr_rpi = recent_rpi_photo(fb) #takes rpi photo of after cut
    return compare_recent_rpi(i, prev_rpi, curr_rpi), curr_rpi

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

def compare_recent_rpi(i, bef_rpi, aft_rpi):
    #Take photo to compare to previous photo before cut
    cwd = os.getcwd()
    # image_path  = os.path.join(cwd, "rpi_images", bef_rpi + "_resized.jpg")
    image_path  = os.path.join(cwd, "rpi_images", bef_rpi)
    bef_rpi = cv2.cvtColor(cv2.imread(image_path, 1), cv2.COLOR_BGR2RGB)


    # image_path  = os.path.join(cwd, "rpi_images", aft_rpi + "_resized.jpg")
    image_path  = os.path.join(cwd, "rpi_images", aft_rpi)
    aft_rpi = cv2.cvtColor(cv2.imread(image_path, 1), cv2.COLOR_BGR2RGB)

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
    batch_prune([(center, target)], overhead, False)

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
    target_list = [((45.29328505595777, 764.53952600395), (155.19716919025666, 741.6428834759711)), ((633.7369980250162, 98.24722843976315), (697.8475971033572, 139.46118499012528)), ((525.4072580645161, 138.53387096774168), (353.61088709677415, 219.17298387096753)), ((1352.8346774193546, 1509.3987903225802), (1293.2318548387095, 1456.8080645161285)), ((1322.3791526239768, 425.4228213769861), (1106.36470871, 571.106981277))]
    print(len(target_list))
    batch_prune_scissors(target_list, args.overhead, args.rpi_check_prune)

    ### External Pot
    # potted_plant_manual(args.overhead)
    
