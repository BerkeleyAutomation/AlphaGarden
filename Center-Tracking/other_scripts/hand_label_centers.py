import cv2
import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import heapq 
from center_constants import *
from centers_test import *

##############################################################################
#To Run these scripts push them into the outer Post-Processing-Scripts folder#
##############################################################################


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
    save_centers("./centers/daily_centers/centers-"+f[9:-3]+"txt", centers)

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
    files = daily_files(IMG_DIR)[8:9]
    
    for _ in range(1):
        centers = []
        for f in files:
            cur_img_path = IMG_DIR + '/' + f
            cur_img_path = "./images/maskonlysnc-200928063000000.png"
            print(f)
            img, img_arr = get_img(cur_img_path)
            # lower_bound, upper_bound = calculate_color_range((226, 50, 170), COLOR_TOLERANCE)
            # Get the image and array with the color of the center isolated
            # plant_img, plant_img_arr = isolate_color(img, lower_bound, upper_bound)
            label_all_centers(img)
        save_centers("./centers/daily_centers/all-centers-" + str(round(centers[0][0])) + "-" + str(round(centers[0][1])) + ".txt", centers)
        