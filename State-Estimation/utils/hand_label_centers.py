import cv2
import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import heapq
from utils.center_constants import *
from utils.centers_test import *
# from full_auto_utils import *
# from full_auto_circles import *

##############################################################################
#To Run these scripts push them into the outer Post-Processing-Scripts folder#
##############################################################################

def daily_files(path, filtered = True):
    ''' returns a list of the first image taken each day in the given folder'''
    file_list = os.listdir(path)
    list.sort(file_list)
    #Only keep files from the same days
    copy_file_list = file_list[:]
    i = 0
    label_prefix = file_list[0].find("-") + 1
    DATE_LENGTH = 6
    while i < len(copy_file_list) and filtered:
        curPrefix = copy_file_list[i][:label_prefix + DATE_LENGTH]
        i = i + 1
        while i < len(copy_file_list) and copy_file_list[i].startswith(curPrefix):
            # os.system("rm "+path+"/"+copy_file_list[i])
            # print("removing: " + path+"/"+copy_file_list[i])
            file_list.remove(copy_file_list[i])
            i = i + 1
    return file_list


centers = []
def label_all_centers(img):
    global centers
    fig, ax = plt.subplots()
    ax.imshow(img)
    def onclick(event):
        centers.append((event.xdata, event.ydata))
        plt.plot(event.xdata, event.ydata, '.', color="w")
        fig.canvas.draw()
        # if len(centers) == 50:
        #     plt.close()
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    save_centers("./centers.txt", centers)

def label_single_center(files):
    global centers
    fig, ax = plt.subplots()
    ax.imshow(img)
    def onclick(event):
        centers.append((event.xdata, event.ydata))
        print((event.xdata, event.ydata))
        plt.plot(event.xdata, event.ydata, '.', color="w")
        fig.canvas.draw()
        plt.close()
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

if __name__ == "__main__":
    files = daily_files(cropped)[8:9]
    for _ in range(1):
        centers = []
        for f in files:
            cur_img_path = IMG_DIR + '/' + f
            cur_img_path = "../../cropped/snc-21052020383300.jpg"
            print(f)
            img, img_arr = get_img(cur_img_path)
            # lower_bound, upper_bound = calculate_color_range((226, 50, 170), COLOR_TOLERANCE)
            # Get the image and array with the color of the center isolated
            # plant_img, plant_img_arr = isolate_color(img, lower_bound, upper_bound)
            label_all_centers(img)
        save_centers("./centers1.txt", centers)
