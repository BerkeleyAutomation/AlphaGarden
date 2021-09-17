from move_farmbot import move_to
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

def farmbot_approach_grids(target, image_name):
    #returns a list x coordinates and a list y coordinates to approach 
    #the target location from the current location in the image
    increment_size = 0.2 # in cm
    epsilon = 0 # the threshold needed to satisfy the closeness requirement
    curr_pos = process_image(image_name)

    x_list = [curr_pos[0] + i*increment_size for i in range(0, int(abs(target[0] - curr_pos[0])/increment_size) + 1)]
    y_list = [curr_pos[1] + i*increment_size for i in range(0, int(abs(target[1] - curr_pos[1])/increment_size) + 1)]

    return x_list[1:], y_list[1:]


def process_image(image_name):
    #finds pixel value of the red dot and then converts that into (x, y) cm position

    cwd = os.getcwd()
    image_path  = os.path.join(cwd, image_name)
    #read image
    src = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # extract red channel
    red_channel = src[:,:,2]
    
    #find indices of max red value
    result = np.where(red_channel == np.amax(red_channel))
    listOfPixels = list(zip(result[0], result[1]))

    curr_pos = process_pixels(listOfPixels)

    return curr_pos


def process_pixels(pixels):
    #put the pixels in bins and return the (x, y) coord with the most pixels
    coor_dict = {}
    for i in range(len(pixels)):
        curr_coor = convert_p_to_c(pixels[i])
        if curr_coor in coor_dict:
            coor_dict[curr_coor] += 1
        else:
            coor_dict[curr_coor] = 1

    max_key = max(coor_dict, key=coor_dict.get)

    return max_key

def convert_p_to_c(pixel_loc):
    #convert the pixels location to the (x, y) closest cm
    #TODO
    return

def convert_c_to_p(coord):
    #convert the (x, y) to the pixels location
    #TODO

    return 


###########################################################################
def local_image_preprocess(local_image):
    #array of the preprocessed local image
    resize_local(local_image) #rotate and rescale

    cwd = os.getcwd()
    image_path  = os.path.join(cwd, "rpi_images", local_image + "_resized.jpg")
    src = cv2.imread(image_path, 1)
    return src

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
    local_image = local_image_preprocess(local_image)
    overhead_image = overhead_image_preprocess(overhead_image)


    img = local_image
    img2 = img.copy()
    template = overhead_image
    w, h = template.shape[:2][::-1]

    meth = 'cv2.TM_CCOEFF_NORMED'
    #150 x 100 y

    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    
    targetpx_x = round((274.66 - target[0])*11.9) + 102
    targetpx_y = round(target[1] * 11.9) + 72
    error = [44, 20] #the border around the target to constrain the region of interest

    sf = 11.9 #scale factor for overhead image ex. 11.9 px = 1 cm

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

    print(masked_res.shape, res_x_lower, res_x_upper, res_y_lower, res_y_upper)
    masked_res[res_y_lower:res_y_upper, res_x_lower:res_x_upper] = res[res_y_lower:res_y_upper, res_x_lower:res_x_upper]
    
    #cv2.imwrite("Masked_res.png", masked_res)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_res)
    print(max_val, max_loc)
    

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    #checking the cross correlation, white = more correlated
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    #plt.show()
    
    (tH, tW) = local_image.shape[:2]

    (startX, startY) = (int(max_loc[0]), int(max_loc[1]))
    (endX, endY) = (int(max_loc[0] + tW), int(max_loc[1] + tH))
    # draw a bounding box around the detected result and display the image
    cv2.rectangle(template, (startX, startY), (endX, endY), (0, 0, 255), 2)

    x_px = (startX + endX) / 2 - 102
    y_px = (startY + endY) / 2 - 72
    pred_pt = (round(274.66 - x_px/11.9), round(y_px/11.9))


    cv2.imwrite(local_name[:-5] + "_" + str(pred_pt) + ".png", template)
    cv2.waitKey(0)

    #(274.66, 0) cm in overhead is (130, 100) px

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

def resize_local(local_image):
    cwd = os.getcwd()
    image_path  = os.path.join(cwd, "rpi_images", local_image)
    img = cv2.imread(image_path, 1)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) #rotate to align with overhead image
     
    scale_factor = 0.7438 # Determined from px to cm calculations from local and overhead images
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # for shrinking INTER_AREA preferred
    plt.imsave(image_path  + "_resized.jpg", resized)

def farmbot_target_approach(fb, target_point, overhead_image):
    #have farmbot apporach the target within same local image
    epsilon = 1 # the threshold needed to satisfy the closeness requirement
    handler = MyHandler("move", (target_point[0] * 10, target_point[1] * 10,0))
    fb.connect(handler)

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
        coord_x += int(np.sign(diff_x) * min(3, np.abs(diff_x)))
        coord_y += int(np.sign(diff_y) * min(3, np.abs(diff_y)))
        handler = MyHandler("move", (coord_x * 10, coord_y * 10,0))
        fb.connect(handler)
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
        handler = MyHandler("move", (coord_x * 10, coord_y * 10,0))
        fb.connect(handler)
    
    return tuple((coord_x, coord_y))

def separate_list(target_list):
    x_list, y_list = [], []
    for i in range(len(target_list)):
        target, center = i[0], i[1]
        if np.abs(target[0] - center[0]) > np.abs(target[1] - center[1]):
            y_list.append(target)
        else:
            x_list.append(target)

    return x_list, y_list

def batch_target_approach(fb, target_list, overhead):
    actual_farmbot_coord = []
    for i in range(len(target_list)):
        #convert target point
        target_point = crop_o_px_to_cm(target_list[i][0][0], target_list[i][0][1]) #assuming each point is (target point, center)
        act_pt = farmbot_target_approach(fb, target_point, args.overhead)
        print(act_pt)
        actual_farmbot_coord.append(act_pt)
    print(actual_farmbot_coord)

    return actual_farmbot_coord


def batch_prune(target_list, overhead, rpi_check):
    fb = FarmBotThread()
    actual_farmbot_coords = batch_target_approach(fb, target_list, overhead)
    x_list, y_list = separate_list(target_list)

    #tool_mounted = ""
    dismount_nozzle()
    mount_xPruner()
    for i in x_list:
        handler = MyHandler("move", (i[0] * 10, i[1] * 10,0))
        fb.connect(handler)
        if rpi_check:
            done = False
            while (done == False):
                #go down z cm prune and come back up
                bef_name = recent_rpi_photo(fb)
        
                handler = MyHandler("prune", None)
                fb.connect(handler)

                aft_name = recent_rpi_photo(fb)
                done = check_prune(bef_name, aft_name)
        else:
            handler = MyHandler("prune", None) #TODO add functionality to go up and down and prune
            fb.connect(handler)
        #prune action

    dismount_xPruner()
    mount_yPruner()

    for i in y_list:
        handler = MyHandler("move", (i[0] * 10 - 40, i[1] * 10 + 40,0)) #y requires offset
        fb.connect(handler)
        if rpi_check:
            done = False
            while (done == False):
                #go down z cm prune and come back up
                bef_name = recent_rpi_photo(fb)
        
                handler = MyHandler("prune", None)
                fb.connect(handler)

                aft_name = recent_rpi_photo(fb)
                done = check_prune(bef_name, aft_name)
        else:
            handler = MyHandler("prune", None) #TODO add functionality to go up and down and prune
            fb.connect(handler)
        #prune action

    dismount_yPruner()
    mount_nozzle()

    return None

def crop_o_px_to_cm(x_px, y_px):
    pred_pt = (round(274.66 - (x_px - 102)/11.9), round((y_px - 72)/11.9))
    return pred_pt

def recent_rpi_photo(fb):
    handler = MyHandler("photo", None)
    fb.connect(handler)
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
    threshold = 0.5 #threshold for normalized ccoeff if pruned or not

    bef_rpi = bef_rpi.copy()
    aft_rpi = aft_rpi.copy()

    method = eval(meth)

    # Apply template Matching

    res = cv2.matchTemplate(bef_rpi, aft_rpi.astype(np.uint8), method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(max_val, max_loc)
    prune = True if max_val < threshold else False
    return prune

def curr_pos_from_local(fb, overhead_image, target):
    cwd = os.getcwd()
    rpi_folder_path = os.path.join(cwd, "rpi_images")
    if not os.path.exists(rpi_folder_path):
        os.makedirs(rpi_folder_path)
    
    #TODO take photo API and place in rpi_images directory
    handler = MyHandler("photo", None)
    fb.connect(handler)

    time.sleep(15)
    photo(rpi_folder_path + "/")


    time.sleep(5)

    list_of_files = glob.glob(rpi_folder_path + '/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)

    print(latest_file)

    local_name = latest_file[latest_file.find("rpi_images")+11:]

    print(local_name)
    pt = find_local_in_overhead(local_name, overhead_image, target)
    print(pt)
    return pt



if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--overhead", "-o", type=str, default="", help="Specify the overhead")
    parser.add_argument("--target_x", "-x", type=int, default=0, help="Specify target x coord")
    parser.add_argument("--target_y", "-y", type=int, default=0, help="Specify target y coord")
    parser.add_argument("--rpi_check_prune", "-p", type=bool, default=False, help="Use rpi images to check if the tool correctly pruned the target leaf")



    args = parser.parse_args()

    im = "snc-21052908141500.jpg_cropped.jpg"
    #print(get_points(im)) #Find Points to crop in overhead image
    #farmbot_target_approach(tuple((args.target_x, args.target_y)), args.overhead)
    target_list = [((2481.782258064516, 923.8887096774195), (2000, 900)), (2250.383064516129, 678.4653225806451), (3098.8467741935483, 1337.6024193548387)]
    #batch_target_approach(target_list, args.overhead)

    batch_prune(target_list, args.overhead, args.rpi_check_prune)
    
