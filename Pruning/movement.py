import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import argparse
import imutils
import math
import glob
from control import start, MyHandler, mount_xPruner, mount_yPruner, dismount_xPruner, dismount_yPruner, mount_nozzle, dismount_nozzle, photo
from thread import FarmBotThread
import argparse
import time
import pickle as pkl

def local_image_preprocess(local_image, sf=0.7438):
    #array of the preprocessed local image
    local_path = resize_local(local_image, sf) #rotate and rescale

    cwd = os.getcwd()
    image_path  = os.path.join(cwd, "rpi_images", local_image + "_" + str(sf) + "_resized.jpg")
    src = cv2.imread(image_path, 1)

    return src, local_path

def overhead_image_preprocess(overhead_image):
    #array of the preprocessed overhead image
    crop_overhead(overhead_image) #crop overhead image

    cwd = os.getcwd()
    image_path  = os.path.join(cwd, overhead_image + "_cropped.jpg")
    src = cv2.imread(image_path, 1)
    return src


def find_local_in_overhead(local_image, overhead_image, target):
    
    #preprocess the overhead image and the raspberry pi local image
    local_name = local_image
    
    overhead_image = overhead_image_preprocess(overhead_image)

    
    template = overhead_image
    w, h = template.shape[:2][::-1]

    meth = 'cv2.TM_CCOEFF_NORMED'
    #150 x 100 y

    method = eval(meth)

    # Apply template Matching
    
    targetpx_x = round((274.66 - target[0])*11.9) + 102
    targetpx_y = round(target[1] * 11.9) + 72
    error = [44, 20] #the border around the target to constrain the region of interest

    best_sf = 1
    best_max_val = 0
    best_max_loc = None
    sf = 11.9 #scale factor for overhead image ex. 11.9 px = 1 cm

    for scale in np.linspace(0.4, 0.85, 15)[::-1]:
        img, rpi_path_d = local_image_preprocess(local_image, scale)
        img2 = img.copy()
        img = img2.copy()
        r = img.shape[1] / float(img.shape[1])


        res = cv2.matchTemplate(img, template.astype(np.uint8) ,method)

        masked_res = np.zeros(res.shape)
        res_x_lower = int(targetpx_x - img.shape[0]/2 - int(error[0]*sf/2))
        res_x_lower = res_x_lower if res_x_lower >=0 else 0
        res_x_upper = int(targetpx_x - img.shape[0]/2 + int(error[0]*sf/2))
        res_x_upper = res_x_upper if res_x_upper <=masked_res.shape[1] else masked_res.shape[1]

        res_y_lower = int(targetpx_y - img.shape[1]/2 - int(error[1]*sf/2))
        res_y_lower = res_y_lower if res_y_lower >=0 else 0
        res_y_upper = int(targetpx_y - img.shape[1]/2 + int(error[1]*sf/2))
        res_y_upper = res_y_upper if res_y_upper <=masked_res.shape[0] else masked_res.shape[0]

        masked_res[res_y_lower:res_y_upper, res_x_lower:res_x_upper] = res[res_y_lower:res_y_upper, res_x_lower:res_x_upper]
            
        _, max_val, _, max_loc = cv2.minMaxLoc(masked_res)

        print(max_val, max_loc, scale)

        if max_val > best_max_val:
            best_max_val = max_val
            best_max_loc = max_loc
            best_sf = scale

        os.remove(rpi_path_d)
        

    top_left = best_max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    img, _ = local_image_preprocess(local_image, best_sf)
    img2 = img.copy()
    img = img2.copy()

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    #checking the cross correlation, white = more correlated
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    #plt.show()
    
    (tH, tW) = img.shape[:2]

    (startX, startY) = (int(best_max_loc[0]), int(best_max_loc[1]))
    (endX, endY) = (int(best_max_loc[0] + tW), int(best_max_loc[1] + tH))
    # draw a bounding box around the detected result and display the image
    cv2.rectangle(template, (startX, startY), (endX, endY), (0, 0, 255), 2)

    x_px = (startX + endX) / 2 - 102
    y_px = (startY + endY) / 2 - 72
    pred_pt = (round(274.66 - x_px/sf), round(y_px/sf))


    cv2.imwrite(local_name[:-5] + "_" + str(pred_pt) + ".png", template)
    cv2.waitKey(0)

    #(274.66, 0) cm in overhead is (102, 72) px

    #Overhead image has 1 cm = 11.9 px

    return pred_pt

def correct_image(im_src, one, two, three, four):
    size = (3478,1630,3) #change this or just take img size
    im_dst = np.zeros(size, np.uint8)
    pts_dst = np.array(
                       [
                        [0,0],
                        [size[0] - 1, 0],
                        [size[0] - 1, size[1] -1],
                        [0, size[1] - 1 ]
                        ], dtype=float
                       )
    pts_src = np.array(
                       [
                        [one[0],one[1]],
                        [two[0], two[1]],
                        [three[0], three[1]],
                        [four[0], four[1]]
                        ], dtype=float
                       )
    h, status = cv2.findHomography(pts_src, pts_dst)
    im_dst = cv2.warpPerspective(im_src, h, size[0:2])
    return im_dst

def get_points(overhead_image):
    #get coords for correct_image from overhead
    cwd = os.getcwd()
    image_path  = os.path.join(cwd, overhead_image)
    im1 = cv2.imread(image_path)
    plt.imshow(im1)
    coords = plt.ginput(4, timeout=0)
    plt.close()
    return coords

def crop_overhead(overhead_image):
    cwd = os.getcwd()
    image_path  = os.path.join(cwd, overhead_image)
    im = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    if im.shape[0] > 3900 or im.shape[1] > 2000:
        im = correct_image(im, (93.53225806451621, 535.8709677419356), (3765.064516129032, 433.2903225806449), (3769.3387096774195, 2241.274193548387), (144.82258064516134, 2241.274193548387))
        plt.imsave(image_path  + "_cropped.jpg", im)

def resize_local(local_image, scale_factor=0.7438):
    # Default scale_factor empirically determined from px to cm calculations from local and overhead images
    cwd = os.getcwd()
    image_path  = os.path.join(cwd, "rpi_images", local_image)
    img = cv2.imread(image_path, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) #rotate to align with overhead image
     
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # for shrinking INTER_AREA preferred
    plt.imsave(image_path  + "_" + str(scale_factor) + "_resized.jpg", resized)
    return image_path  + "_" + str(scale_factor) + "_resized.jpg"

def farmbot_target_approach(fb, target_point, overhead_image, y_offset):
    #have farmbot apporach the target within same local image
    epsilon = 1 # the threshold needed to satisfy the closeness requirement
    max_y = (125-abs(y_offset))
    # print("HERE: ", max_y)
    fb.update_action("move", (target_point[0] * 10, min(target_point[1] * 10, max_y*10), 0)) #target_point[1] * 10,0))
    time.sleep(30)

    curr_pos = curr_pos_from_local(fb, overhead_image, target_point)#get from local image
        
    coord_x = target_point[0]
    coord_y = target_point[1]
    previous_points = []
    count = 0
    while ((np.linalg.norm(np.array(target_point) - np.array(curr_pos)) > epsilon) and count <= 6): #add 6 iteration limit, average last three
        curr_x, curr_y  = curr_pos[0], curr_pos[1]
        diff_x = int(target_point[0] - curr_x)
        diff_y = int(target_point[1] - curr_y)  #increment with a vector

        print(diff_x, diff_y)
        coord_x += int(np.sign(diff_x) * min(4, np.abs(diff_x)))
        coord_y += int(np.sign(diff_y) * min(4, np.abs(diff_y)))
        # Cap to limit movement error
        coord_x = min(coord_x, 274)
        coord_x = max(0, coord_x)
        if coord_y > (125-abs(y_offset)):
            print("------CAPPED------")
        coord_y = min(coord_y, (125-abs(y_offset))) #125
        coord_y = max(0, coord_y)

        fb.update_action("move", (coord_x * 10, coord_y * 10,0))
        time.sleep(3)

        curr_pos = curr_pos_from_local(fb, overhead_image, target_point)#get from local image
        count += 1
        previous_points.append(tuple((coord_x, coord_y)))
    if count >= 6:
        coord_x = int(np.mean([i[0] for i in previous_points[-3:]]))
        coord_y = int(np.mean([i[1] for i in previous_points[-3:]]))
    response = input("Enter 'y' if ready to prune or 'n' if not: ")
    if response == "n":
        x = int(input("Enter x adjustment (cm): "))
        y = int(input("Enter y adjustment (cm): "))
        coord_x += x
        coord_y += y
        fb.update_action("move", (coord_x * 10, coord_y * 10,0))
    
    return tuple((coord_x, coord_y))

def batch_target_approach(fb, target_list, overhead, y_offset):
    actual_farmbot_coord = []
    for i in range(len(target_list)):
        #convert target point
        target_point = crop_o_px_to_cm(target_list[i][1][0], target_list[i][1][1]) #assuming each point is (center point, target)
        act_pt = farmbot_target_approach(fb, target_point, overhead, y_offset[i])
        actual_farmbot_coord.append(act_pt)
        pkl.dump(actual_farmbot_coord, open("actual_coords.p", "wb"))
    return actual_farmbot_coord

def crop_o_px_to_cm(x_px, y_px):
    #convert pixel to cm in overhead image with set scale factor of 11.9
    pred_pt = (round(274.66 - (x_px - 102)/11.9), round((y_px - 72)/11.9))
    return pred_pt


def curr_pos_from_local(fb, overhead_image, target):
    cwd = os.getcwd()
    rpi_folder_path = os.path.join(cwd, "rpi_images")
    if not os.path.exists(rpi_folder_path):
        os.makedirs(rpi_folder_path)
    
    fb.update_action("photo", None)

    time.sleep(15)
    photo(rpi_folder_path + "/")

    time.sleep(5)

    list_of_files = glob.glob(rpi_folder_path + '/*')
    latest_file = max(list_of_files, key=os.path.getctime)

    local_name = latest_file[latest_file.find("rpi_images")+11:]

    pt = find_local_in_overhead(local_name, overhead_image, target)
    return pt
