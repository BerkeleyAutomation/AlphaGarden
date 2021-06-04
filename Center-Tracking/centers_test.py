import cv2
import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import heapq
import pandas as pd
import datetime as dt
from datetime import timedelta
import os
# import seaborn as sns
from center_constants import *
from geometry_utils import *



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

def calculate_updated_center(old_center, mask_center):
    '''Find the new calculated center given the old center and the calculated mask center
    current implementations finds the midpoint
    '''
    #TODO add weighting/certainty to this calculation
    return (old_center[0] + mask_center[0]) / 2, (old_center[1] + mask_center[1]) / 2

def not_black(point, img_arr):
    '''Checks that a pixel isn't black'''
    rgb = img_arr[point[1]][point[0]]
    return rgb[0] > 100 or rgb[1] > 100 or rgb[2] > 100

def valid_point(point, img_arr, visited = set()):
    '''helper function for find_color'''
    return point[0] >= 0 and point[1] >= 0 \
        and point[0] < img_arr.shape[1] and point[1] < img_arr.shape[0] \
            and point not in visited

def neighbors(point, img_arr, visited):
    '''helper function for find_color'''
    delta = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    neighs = []
    for d in delta:
        neighbor = (point[0] - d[0], point[1] - d[1])
        if valid_point(neighbor, img_arr, visited):
            visited.add(neighbor)
            neighs.append(neighbor)
    return neighs

def find_color(center, img_arr):
    '''Uses BFS to find the color associated with the current center.
    Assumes color closest to the center is the correct color
    returns the RGB of the color.
    '''
    visited = set()

    def distance_to_center(p1):
        return distance(p1, center)

    def is_valid_color(point):
        temp_rbg = img_arr[round(point[1])][round(point[0])]
        temp_rbg = (temp_rbg[0], temp_rbg[1], temp_rbg[2])
        return temp_rbg in COLORS

    q = [(0, center)]
    while q:
        cur_point = heapq.heappop(q)[1]
        visited.add(cur_point)
        if is_valid_color(cur_point):
            found = img_arr[cur_point[1]][cur_point[0]]
            return min(COLORS, key = lambda c: sum([abs(c - s) for s,c in zip(c, found)])), cur_point
        for n in neighbors(cur_point, img_arr, visited):
            heapq.heappush(q, (distance_to_center(n), n))
    return (0, 0, 0)

def mask_center_of_mass(old_center, img_arr, radius = 100):
    '''Gets the center of mass of the plant in the img_arr radius around old_center in img_arr'''
    total_x, total_y, count = 0, 0, 0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            cur_p = old_center[0] - i, old_center[1] - j
            if not valid_point(cur_p, img_arr):
                continue
            if not_black(cur_p, img_arr):
                total_x += cur_p[0]
                total_y += cur_p[1]
                count += 1
    return (total_x / count, total_y / count) if count > 0 else old_center

def update_min_max(point, max_x, max_y, min_x, min_y):
    max_x = max(max_x, point, key = lambda p: p[0])
    max_y = max(max_y, point, key = lambda p: p[1])
    min_x = min(min_x, point, key = lambda p: p[0])
    min_y = min(min_y, point, key = lambda p: p[1])
    return max_x, max_y, min_x, min_y

def mask_center_by_max_radius(old_center, img_arr, radius = 100):
    '''Returns a circle with radius equal to the major axis of the plant. The circle is returned
    as a tuple of the center and the radius -> Tuple((x, y), r)'''

    max_x, max_y, min_x, min_y = (-float("inf"), -float("inf")), (-float("inf"), -float("inf")), \
        (float("inf"), float("inf")), (float("inf"), float("inf"))
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            cur_p = old_center[0] - i, old_center[1] - j
            if not valid_point(cur_p, img_arr):
                continue
            if not_black(cur_p, img_arr):
                max_x, max_y, min_x, min_y = update_min_max(cur_p, max_x, max_y, min_x, min_y)
    max_pair = max((max_x, min_x), (max_y, min_y), key=lambda t: sq_distance(t[0], t[1]))
    min_pair = min((max_x, min_x), (max_y, min_y), key=lambda t: sq_distance(t[0], t[1]))
    return find_circle(max_pair, min_pair[0], min_pair[1])

def get_img(path):
    '''Using the full path, get an image and return it's RGB image and RGB array'''
    bgr_img = cv2.imread(path)
    print(path)
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img_arr = np.asarray(img)
    return img, img_arr

def isolate_color(img, lower_bound, upper_bound):
    '''Isolate all pixels of color between the bounds and return that image and corresponding array'''
    mask = cv2.inRange(img, lower_bound, upper_bound)
    true_color_mask = cv2.bitwise_and(img, img, mask=mask)
    return true_color_mask, np.asarray(true_color_mask)

def calculate_color_range(rgb, tolerance):
    '''Returns a tolerance range for the color, because there is some fluxuation'''
    lower_bound = np.array([max(val - tolerance, 0) for val in rgb])
    upper_bound = np.array([min(val + tolerance, 255) for val in rgb])
    return lower_bound, upper_bound

def save_centers(path, centers):
    '''Save centers list at the path (must be .txt)'''
    with open(path, 'w') as f:
        for listitem in centers:
            f.write('%s\n' % str(listitem))

def read_centers(path):
    '''get centers from text file'''
    centers = []
    with open(path, 'r') as f:
        for line in f:
            cur_center = line[:-1]
            centers.append(eval(cur_center))
    return centers



def calculate_error(predicted, actual):
    errors = []
    for p, a in zip(predicted, actual):
        errors.append(sq_distance(p, a))
    return errors

def find_new_center(old_center, img_arr, img, radius = 100, predicted = [0, 0]):
    '''Wrapper function that finds and returns the model's
     estimate of a center given old center and the image's
     numpy array'''
    # Get color information and tolerances
    rgb_center, first_color_pixel = find_color(old_center, img_arr)
    lower_bound, upper_bound = calculate_color_range(rgb_center, COLOR_TOLERANCE)
    # Get the image and array with the color of the center isolated
    plant_img, plant_img_arr = isolate_color(img, lower_bound, upper_bound)
    # Calculate the plant's center using center of maxx
    mask_center = mask_center_of_mass(first_color_pixel, plant_img_arr, radius)
    mask_center = calculate_updated_center(old_center, mask_center)
    #Extrema Center
    mask_center2, r = mask_center_by_max_radius(first_color_pixel, plant_img_arr, radius)
    #Static center
    mask_center3 = old_center
    fig, ax = plt.subplots()
    ax.imshow(plant_img)
    plt.plot(mask_center[0], mask_center[1], 'o', color="w")
    plt.plot(predicted[0], predicted[1], 'o', color="y")
    plt.plot(mask_center2[0], mask_center2[1], 'o', color="c")
    plt.plot(mask_center3[0], mask_center3[1], 'o', color="g")
    # circle1 = Circle((mask_center2[0], mask_center2[1]), r, color='b', fill=False, lw=3)
    # ax.add_patch(circle1)
    fig.canvas.draw()
    plt.show()

    return mask_center

def track_one_plant(init_center, predicted_centers):
    '''Mainly for testing purpose:
    Calculate the center of one plane with the inital center given then calculate error
    '''
    center = init_center
    files = daily_files(IMG_DIR)
    centers = []
    radius = 100
    for i, day in enumerate(files):
        print(day)
        cur_img_path = IMG_DIR + '/' + day
        img, img_arr = get_img(cur_img_path)
        # Get center of the plant
        center = find_new_center(center, img_arr, img, radius, predicted_centers[i])
        print("center: ", center)
        centers.append(center)
        center = (round(center[0]), round(center[1]))
        radius = min(radius+10, 200)
    centers = pixels_to_cm(centers)
    predicted_centers = pixels_to_cm(predicted_centers)
    return calculate_error(predicted_centers, centers)

def track_all_plants(init_centers, img_path):
    ''' Given tuples of inital centers of the form (x: float, y: float) update and
    save centers for all given current centers.
    '''
    img, img_arr = get_img(img_path)
    updated_centers = []
    for center in init_centers:
        new_center = find_new_center(center, img_arr, img)
        updated_centers.append(new_center)
    save_centers(updated_centers)


if __name__ == "__main__":
    # predicted = read_centers("./centers/daily_centers/centers-3098.8467741935483-117.49758064516118.txt")
    # dist_off = track_one_plant((round(predicted[0][0]), round(predicted[0][1])), predicted)
    # save_centers("./centers/com_centers.txt", dist_off)
    com = read_centers("./centers/com_centers.txt")
    circle = read_centers("./centers/circle_centers.txt")
    static = read_centers("./centers/static_centers.txt")
    start_date = dt.date(2020, 9, 20)
    end_date = dt.date(2020, 10, 17)
    dates = [str(dt.date.fromordinal(i)) for i in range(start_date.toordinal(), end_date.toordinal())]
    sns.set_theme(style="darkgrid")
    sns.lineplot(x=dates, y=static, label = "Static")
    sns.lineplot(x=dates, y=com, label = "Center of Mass")
    sns.lineplot(x=dates, y=circle, label = "Circle")
    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.2))
    # plt.plot(dates, dist_off)
    plt.title("Error from 9/20-10/17")
    plt.xlabel("Date")
    plt.ylabel("Distance ground-truth center (cm)")
    print([date for i, date in enumerate(dates) if i % 6 == 0])
    plt.xticks(np.arange(0, len(dates)+1, 5), rotation=45)
    plt.show()
