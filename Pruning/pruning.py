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
    actual_farmbot_coords = batch_target_approach(fb, target_list, overhead, offset)
    # actual_farmbot_coords = [(209, 17), (251, 68)]
    print("--ACTUAL FARMBOT COORDS: ", actual_farmbot_coords)
    height_fb_clearance = 8 #cm from top of farmbot

    for cur_point, angle in zip(actual_farmbot_coords, angles):
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
        response = input("===== Enter 'y' when READY to prune.")

        fb.update_action("move_rel", (0, 0, (z * -10)+ 50))#move to z position from the depth sensor after setting up the scissors
        time.sleep(90)
        print("---TIME TO CUT")
        done = False
        if False: #z < height_fb_clearance:
            #cut top off - NOT IMPLEMENTED
            fb.update_action("servo", (6, 101)) #determine the pin
            while (done == False):
                fb.update_action("prune_scissor", None) #prune with angle
                time.sleep(11)
                done = prune_check_sensor(fb, z, dsensor_adjusted, scissors_offset)
        else:
            #radial cut with certain angle
            fb.update_action("servo", (6, 38)) #determine the pin
            time.sleep(2)
            while (done == False):
                fb.update_action("prune_scissor", None) #prune with angle
                time.sleep(11)
                fb.update_action("move_rel", (0, 0, (z * 10) - 50.5))#move to z position from the depth sensor after setting up the scissors
                time.sleep(90)
                done = prune_check_sensor(fb, z, dsensor_adjusted, scissors_offset)
                print("CHECK: ", done)
                done = True
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


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--overhead", "-o", type=str, default="", help="Specify the overhead")
    parser.add_argument("--rpi_check_prune", "-p", type=bool, default=False, help="Use rpi images to check if the tool correctly pruned the target leaf")

    args = parser.parse_args()

    # Initialize - comment out after failure
    # pkl.dump([], open("actual_coords.p", "wb"))

    # target_list = [((2481.782258064516, 923.8887096774195), (2000, 900)), ((2250.383064516129, 678.4653225806451), (2100, 600)), ((3098.8467741935483, 1337.6024193548387), (3000, 1000))]
    # target_list = [((2192.1348713372863, 1033.2203931840422), (2051.2697426745726, 766.4407863680847)), ((2646.372097874638, 677.2447216554123), (2762.744195749276, 475.4894433108245)), ((2469.969855260587, 922.4155430868436), (2405.9397105211738, 680.8310861736873)), ((2469.4256064016004, 993.9150318829708), (2579.8512128032007, 965.8300637659415)), ((2358.408583699892, 1373.3575856917587), (2478.817167399784, 1297.7151713835174)), ((2533.076888573078, 1331.9534684904738), (2850.1537771461562, 1248.9069369809476)), ((2211.2047908751406, 1345.0958580073393), (2084.409581750281, 1262.1917160146788)), ((3174.827852473074, 860.0234503457663), (2972.6557049461485, 877.0469006915325))]
    # target_list = [((2277.5, 812.0), (2406, 681)), ((2701.746661309253, 1078.2378295073647), (2605.493322618506, 976.4756590147294)), ((2154.69379587411, 962.8057578959075), (2045.3875917482205, 762.611515791815)), ((2676.7439679713843, 364.66091015743575), (2773.4879359427687, 472.3218203148715)), ((2534.407785238923, 1414.590281191918), (2509.8155704778464, 1313.1805623838359)), ((3145.0094609449748, 1407.244076094701), (2878.0189218899495, 1270.4881521894022)), ((2105.8164770134176, 1122.0139410350807), (2080.6329540268353, 1266.0278820701615)), ((3104.0017608655576, 950.9779367359782), (2978.003521731115, 876.9558734719565)), ((3153.5, 1509.0), (3113, 1409))]
    #target_l = pkl.load(open("/Users/mpresten/Desktop/AlphaGarden_git/AlphaGarden/Center-Tracking/current_pts.p", "rb"))
    #print(target_l)
    # target_list = [[(2404.649193548387, 734.5620967741932), (2180.262096774193, 734.5620967741932)]] # should be a y cut
    # target_list = [[(2397.637096774193, 695.9955645161285), (2404.649193548387, 955.44314516129)]] #should be a an x cut
    # target_list = [[(2411.6612903225805, 717.0318548387093), (2236.358870967742, 909.8645161290319)]] #scissors should point towards neg x pos y
    # target_list = [[(2373.094758064516, 727.5499999999995), (2586.963709677419, 937.9129032258061)]] #scissors should point towards pos x pos y 

    #borage
    # target_list = [((961.7344893566133, 404.5748624730926), (849, 383)), ((419.1007083074557, 913.2629979593647), (381, 961))]
    #kale and turnip
    target_list = [((671.3921207335263, 696.0703255242609), (452, 701))] # ((1083.3915387737638, 1209.318982634356), (1088, 1049)), ((1329.521974901467, 446.7697818589884), (1206,628))


    #110, 50, 0, 90
    # first = [((2418.9024915528416, 708.3370823083476), (2236.951245776421, 771.1685411541738)), ((2614.5830485467905, 1059.5356036410094), (2520.791524273395, 950.7678018205047)), ((2092.8935432563703, 1282.7822196370712), (2210.946771628185, 1397.8911098185356))]
    #angles = perpendiculars(target_list)
    batch_prune_scissors(target_list, args.overhead, args.rpi_check_prune)
    
