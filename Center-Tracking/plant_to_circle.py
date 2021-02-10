import cv2
import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
from itertools import combinations 
import heapq 
import pandas as pd
import datetime as dt
from collections import deque
from datetime import timedelta
import os
import seaborn as sns
from center_constants import *
from geometry_utils import *
from centers_test import *

#####################################
########## UTILITY METHODS ##########
#####################################


def draw_circles(path, centers, radii, circle_color = "y"):
    '''Draws the circle that corresponds to the plants with centers and radii
    circles and radii should have equal length and correspond with eachother 
    '''
    # print(centers, radii)
    img, img_arr = get_img(path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    for center, radius in zip(centers, radii):
        circle1 = Circle((round(center[0]), round(center[1])), radius, color=circle_color, fill=False, lw=2)
        ax.add_patch(circle1)
    plt.show()

def draw_circle_sets(path, centers, list_of_radii, colors):
    assert len(colors) == len(list_of_radii)
    img, img_arr = get_img(path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    for radii, color in zip(list_of_radii, colors):
        for center, radius in zip(centers, radii):
            circle = Circle((round(center[0]), round(center[1])), radius, color=color, fill=False, lw=2)
            ax.add_patch(circle)
    plt.savefig("./figures/circle_comparison.png")
    plt.show()


def convert_to_plant_colorspace(old_center, img_arr):
    rgb_center, first_color_pixel = find_color(old_center, img_arr)
    lower_bound, upper_bound = calculate_color_range(rgb_center, COLOR_TOLERANCE)
    # Get the image and array with the color of the center isolated
    plant_img, plant_img_arr = isolate_color(img, lower_bound, upper_bound)
    return plant_img, plant_img_arr
  
def plant_COM_extreme_points(old_center, img_arr, radius = 100):
    total_x, total_y, count = 0, 0, 0
    max_x, max_y, min_x, min_y = (-float("inf"), -float("inf")), (-float("inf"), -float("inf")), \
        (float("inf"), float("inf")), (float("inf"), float("inf"))
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            cur_p = old_center[0] - i, old_center[1] - j
            if not valid_point(cur_p, img_arr):
                continue
            if not_black(cur_p, img_arr):
                total_x += cur_p[0]
                total_y += cur_p[1]
                count += 1    
                max_x, max_y, min_x, min_y = update_min_max(cur_p, max_x, max_y, min_x, min_y)       
    return ((total_x / count, total_y / count) if count > 0 else old_center), (max_x, max_y, min_x, min_y)  

def bfs(center, img_arr, termination_cond=lambda arr, recent_color_pts, p: len(arr) > 600, arr_oversided = lambda arr, r: 6.3*r < len(arr)):
    visited = set()
    color_pts = set()
    recent_color_pts = 0
    def distance_to_center(p1):
        return sq_distance(p1, center)

    def is_valid_color(point):
        temp_rbg = img_arr[round(point[1])][round(point[0])]
        temp_rbg = (temp_rbg[0], temp_rbg[1], temp_rbg[2])
        return temp_rbg in COLORS
    benchmark_r = 10
    q = [(0, center)]
    recent = deque([1])
    while q:
        current_r, cur_point = heapq.heappop(q)
        benchmark_r = max(benchmark_r, current_r)
        visited.add(cur_point)
        if not_black(cur_point, img_arr):
            color_pts.add(cur_point)
            recent.append(1)
            recent_color_pts += 1
        else:
            recent.append(0)
        if arr_oversided(recent, benchmark_r):
            color_pt = recent.popleft()
            if color_pt:
                recent_color_pts -= 1
        if termination_cond(recent, recent_color_pts, cur_point):
            break
        for n in neighbors(cur_point, img_arr, visited):
            heapq.heappush(q, (distance_to_center(n), n))  
    return color_pts

def get_models():
    return pkl.load(open(RADIUS_MODELS_PATH, "rb" ))

def linearity_score(center, img_arr, radius=100):
    '''
    Returns the ratio of the sides of the rectangle formed by it's extreme points of the plant centered at center
    '''
    center = (round(center[0]), round(center[1]))
    _, extrema = plant_COM_extreme_points(center, img_arr, 100)
    max_pair = max(combinations(extrema, 2), key=lambda p: distance(p[0], p[1]))
    extrema = list(extrema)
    [extrema.remove(p) for p in max_pair]
    print(distance(extrema[0], extrema[1]), distance(max_pair[0], max_pair[1]), max_pair, extrema)
    return distance(extrema[0], extrema[1]) / distance(max_pair[0], max_pair[1])

def approximate_circle_contour(mask, hull):
    hull = np.squeeze(hull, axis=1)
    
    num_points = len(hull)
    centroid_x, centroid_y = 0, 0
    
    # Find the centroid for the hull.
    for point in hull:
        centroid_x += point[0]
        centroid_y += point[1]
    centroid_x //= num_points
    centroid_y //= num_points
    centroid = (centroid_x, centroid_y)
    
    l2_dist = lambda pt1, pt2: ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    l2_dist_higher_order = lambda pt2: lambda pt1: ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    
    # Find the point that is farthest away from the centroid.
    max_radius_point = max(hull, key=l2_dist_higher_order(centroid))
    max_radius = l2_dist(max_radius_point, centroid)

    return (centroid, max_radius)

def merge_circles(circles):
    circles.sort(key=lambda pair: -pair[1])
    circles = set(circles)
    merged = set()
    while len(circles) > 0:
        curr_circle = list(circles)[0]
        circles.remove(curr_circle)
        
        circles_copy = set(circles)
        
        for smaller_circle in circles_copy:
            if lie_within(curr_circle, smaller_circle):
                circles.remove(smaller_circle)
        merged.add(curr_circle)
    return merged

def prepare_binary_mask(plant_type, mask):
    color = TYPES_TO_COLORS[plant_type]
    offset = 5
    indices = np.where(np.all(np.abs(mask - np.full(mask.shape, color)) <= offset, axis=-1))
    coordinates = zip(indices[0], indices[1])
    binary_mask = np.full(mask.shape, (0, 0, 0))
    for coord in coordinates:
        binary_mask[coord[0], coord[1]] = [255, 255, 255]
    binary_mask = binary_mask.astype(np.uint8)
    return binary_mask

def clear_canvas(canvas):
    indices = np.where(np.all(np.abs(canvas - np.full(canvas.shape, [0, 0, 0])) <= 3, axis=-1))
    coordinates = zip(indices[0], indices[1])
    for coord in coordinates:
        canvas[coord[0], coord[1]] = [255, 255, 255]
    return canvas

def lie_within(c1, c2, offset=50): 
    x1, x2 = c1[0][0], c2[0][0]
    y1, y2 = c1[0][1], c2[0][1]
    r1, r2 = c1[1], c2[1]
    distSq = ((x1 - x2)**2+ (y1 - y2)**2)**(0.5) 
    return distSq <= r1 - r2 + offset

############################################
########## CENTER FINDING METHODS ##########
############################################

def bfs_circle(path, old_center, radius=100):
    '''Uses BFS and a termination condition to find the plant.'''
    def termination_cond(arr, recent_color_pts, point): 
        plant_ended = recent_color_pts / len(arr) < .1 and distance(point, old_center) > 20
        too_long = distance(point, old_center) > radius
        return plant_ended or too_long

    img, img_arr = get_img(path)
    img, img_arr = convert_to_plant_colorspace(old_center, img_arr)
    color_points = bfs(old_center, img_arr, termination_cond)
    sum_x, sum_y, extreme_pt = 0, 0, old_center
    for p in color_points:
        sum_x += p[0]
        sum_y += p[1]
        if distance(p, old_center) > distance(extreme_pt, old_center):
            extreme_pt = p
    return (sum_x / len(color_points), sum_y / len(color_points)), distance(extreme_pt, old_center)


def extreme_points_circle(path, old_center, radius = 100):
    '''Finds the circle that corresponds to the plant at old_center using extreme points to create a circle '''
    img, img_arr = get_img(path)
    img, img_arr = convert_to_plant_colorspace(old_center, img_arr)
    return mask_center_by_max_radius(old_center, img_arr, radius)

def max_COM_radius(path, old_center, radius = 100):
    '''
    Finds the circle that corresponds to the plant at old_center by taking the centroid as the center
    and the distance to the farthest extreme point as the radius
    '''
    img, img_arr = get_img(path)
    img, img_arr = convert_to_plant_colorspace(old_center, img_arr)
    center, (max_x, max_y, min_x, min_y) = plant_COM_extreme_points(old_center, img_arr, radius)
    return center, distance(max(max_x, max_y, min_x, min_y, key=lambda p: distance(p,center)), center)

def avg_COM_radius(path, old_center, radius = 100):
    '''
    Finds the circle that corresponds to the plant at old_center by taking the centroid as the center
    and the average distances to the extreme points as the radius
    '''
    img, img_arr = get_img(path)
    img, img_arr = convert_to_plant_colorspace(old_center, img_arr)
    center, (max_x, max_y, min_x, min_y) = plant_COM_extreme_points(old_center, img_arr, radius)
    return center, sum([distance(p,center) for p in [max_x, max_y, min_x, min_y]]) / 4

def min_COM_radius(path, old_center, radius = 100):
    '''
    Finds the circle that corresponds to the plant at old_center by taking the centroid as the center
    and the distance to the CLOSEST extreme point as the radius
    '''
    img, img_arr = get_img(path)
    img, img_arr = convert_to_plant_colorspace(old_center, img_arr)
    center, (max_x, max_y, min_x, min_y) = plant_COM_extreme_points(old_center, img_arr, radius)
    return center, distance(min(max_x, max_y, min_x, min_y, key=lambda p: distance(p,center)), center)

def convert_from_shape_to_circle(path):
    ''' Get's mask circles using convex contours of the plants'''
    mask, img_arr = get_img(path)
    plant_circles = {}
    for plant_type in TYPES_TO_COLORS:
        binary_mask = prepare_binary_mask(plant_type, mask)
        gray_scaled_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(binary_mask)
        contours, hierarchy = cv2.findContours(gray_scaled_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Invert the mask for clearer outline of the convex hull.
        circles = []
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            pair = approximate_circle_contour(inverted, hull)
            circles.append(pair)
        merged = merge_circles(circles)
        merged = [circle for circle in merged if circle[1] >= 10]
        plant_circles[plant_type] = merged
        
    canvas = np.full(mask.shape, [255, 255, 255]).astype(np.uint8)
    for plant_type in plant_circles:
        color = TYPES_TO_COLORS[plant_type]
        for plant_circle in plant_circles[plant_type]:
            centroid = plant_circle[0]
            radius = plant_circle[1]
            canvas = cv2.circle(inverted, centroid, 2, tuple(color), thickness=-1)
            canvas = cv2.circle(inverted, centroid, int(radius), tuple(color), thickness=2)
        canvas = clear_canvas(canvas)
    return canvas, plant_circles


######################################
########## ACCURACY METRICS ##########
######################################

def avg_fill_ratio(centers, radii, path):
    img, img_arr = get_img(path)
    color_ratios = []
    for c, r in zip(centers, radii):
        if r == float("inf") or r == None   :
            continue
        total_points, color_points = 0, 0
        for i in range(-int(r), int(r)+1):
            dx = int((r**2 - i**2)**.5)
            for j in range(-dx, dx):
                cur_p = int(c[0]+i), int(c[1]+j)
                if valid_point(cur_p, img_arr):
                    total_points += 1
                    if not_black(cur_p, img_arr):
                        color_points += 1
        color_ratios.append(color_points / total_points)
    return sum(color_ratios) / len(color_ratios)

def get_total_plant_area(c, max_r, img_arr, condition = lambda x, y: True):
    ''' Gets the total plant area based on an optional condition (lambda x, y: f(x,y)) inside of the circle 
    defined by c, max_r'''
    color_count = 0
    c = (round(c[0]), round(c[1]))
    max_r = int(max_r)
    for i in range(-max_r, max_r):
        for j in range(-max_r, max_r):
            cur_p = (round(c[0] + i), round(c[1] + j))
            if valid_point(cur_p, img_arr) and not_black(cur_p, img_arr) \
                and condition(cur_p[0], cur_p[1]):
                color_count += 1
    return color_count

def avg_circle_to_plant_area_ratio(centers, radii, path):
    color_ratios = []
    original_img, original_img_arr = get_img(path)
    for c, r in zip(centers, radii):
        c = (round(c[0]), round(c[1]))
        img, img_arr = convert_to_plant_colorspace(c, original_img_arr)
        # Find the max viable radius for this plant
        com, max_r = max_COM_radius(path, c, 120)
        max_r = int(max_r)
        color_count = 0
        for i in range(-max_r, max_r):
            for j in range(-max_r, max_r):
                cur_p = (round(c[0] + i), round(c[1] + j))
                if valid_point(cur_p, img_arr) and not_black(cur_p, img_arr):
                    color_count += 1
        color_ratios.append(color_count / (math.pi * r**2))
    return sum(color_ratios) / len(color_ratios)      
        
def avg_excluded_plant_area(centers, radii, path):
    color_ratios = []
    original_img, original_img_arr = get_img(path)
    for c, r in zip(centers, radii):
        c = (round(c[0]), round(c[1]))
        img, img_arr = convert_to_plant_colorspace(c, original_img_arr)
        # Find the max viable radius for this plant
        com, max_r = max_COM_radius(path, c, 140)
        excluded_plant = get_total_plant_area(c, max_r, img_arr, lambda x, y: distance((x,y), c) > r)
        total_plant = get_total_plant_area(c, max_r, img_arr)
        try:
            color_ratios.append(excluded_plant / total_plant)
        except ZeroDivisionError:
            continue
    return sum(color_ratios) / len(color_ratios)    

def simulate_prune(center, radius, reduction, path):
    original_img, original_img_arr = get_img(path)
    img, img_arr = convert_to_plant_colorspace(c, original_img_arr)
    com, max_r = max_COM_radius(path, c, radius + 20)
    prev_area = get_total_plant_area(center, max_r, img_arr)
    pruned_area = get_total_plant_area(center, (1-reduction)*radius, img_arr)
    return prev_plant_area, pruned_area
    




if __name__ == "__main__":
    files = daily_files(IMG_DIR)[8:9]
    centers, max_radii, avg_radii, min_radii = [], [], [], []
    cur_img = IMG_DIR + "/" + files[0]
    # print(cur_img)
    old_centers = read_centers("./centers/daily_centers/all-centers-3070.0-104.0.txt")
    img, img_arr = get_img(cur_img)
    # circles, mapping = convert_from_shape_to_circle(cur_img)
    # for val in mapping.values():
    #     if len(val) > 0:
    #         for c, r in val:
    #             centers.append(c)
    #             max_radii.append(r)
    # draw_circles(cur_img, centers, max_radii)
    for center in old_centers:
        center = (round(center[0]), round(center[1]))
        try:
            # c, r_min = min_COM_radius(cur_img, center, 130)
            # c, r_max = max_COM_radius(cur_img, center, 130)
            # c, r_max = avg_COM_radius(cur_img, center, 130)
            c, r_max = bfs_circle(cur_img, center, 130)
            if r_max == r_max:
                print(c, r_max)
                centers.append(c)
                max_radii.append(r_max) 
                # min_radii.append(r_min) 
                # avg_radii.append(r_avg) 

        except ZeroDivisionError:
            print(center)
    draw_circle_sets(cur_img, centers, [max_radii, min_radii, avg_radii], ("w", 'y', 'r'))
    # print("Avg fill ratio: " + str(avg_fill_ratio(centers, max_radii, cur_img)))
    # print("Plant area to circle area ratio: " + str(avg_circle_to_plant_area_ratio(centers, max_radii, cur_img)))
    # print("Avg excluded plant area: " + str(avg_excluded_plant_area(centers, max_radii, cur_img)))
