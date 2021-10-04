import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import KDTree
from center_constants import *
from geometry_utils import *
from centers_test import *
from plant_to_circle import *

##############################################################################
#To Run these scripts push them into the outer Post-Processing-Scripts folder#
##############################################################################

def get_plant_type(center, img_arr):
    rgb_center, first_color_pixel = find_color(center, img_arr)
    return COLORS_TO_TYPES[rgb_center]

def get_nearby_contours(contours, center, inverted, radius):
    nearby = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        pair = approximate_circle_contour(inverted, hull)
        if distance(center, pair[0]) < radius:
            nearby.append(cnt)
    return nearby

def get_edge_points(center, img, img_arr, radius):
    plant_type = get_plant_type(center, img_arr)
    binary_mask = prepare_binary_mask(plant_type, img)
    gray_scaled_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(binary_mask)
    contours, hierarchy = cv2.findContours(gray_scaled_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = get_nearby_contours(contours, center, inverted, radius)
    return contours

def find_max_extrema(center, img, img_arr, edges, radius):
    max_extreme = center
    for cnt in edges:
        cnt = np.squeeze(cnt, axis=1)
        for point in cnt:
            if sq_distance(point, center) > sq_distance(max_extreme, center):
                max_extreme = point
    return max_extreme

def get_angle_between(p_new, p_ref, center):
    '''Get's the angle created by the two points given and the center'''
    v_ref = [p_ref[0] - center[0], p_ref[1] - center[1]]
    v_new = [p_new[0] - center[0], p_new[1] - center[1]]
    m_ref, m_new = np.divide(v_ref[1], v_ref[0]), np.divide(v_new[1], v_new[0])
    cos_val = np.dot(v_ref, v_new) / (np.linalg.norm(v_ref) * np.linalg.norm(v_new))
    cos_val = max(-1, min(cos_val, 1))
    angle_magnitude = math.acos(cos_val)
    # angle_direction = np.sign(math.atan((m_new-m_ref) / (1-m_ref*m_new)))
    angle_direction = np.sign(v_ref[0]*v_new[1]-v_ref[1]*v_new[0])
    # if angle_magnitude <= math.acos(0):
    return angle_magnitude*angle_direction
    # else:
    #     return -1*angle_magnitude*angle_direction


def get_next_extrema(center, edges, prev_extrema, period_shift = 1):
    '''gets the next extrema'''
    get_angle_HOF = lambda p: get_angle_between(p, prev_extrema, center)
    new_extreme = center
    prev_max_dist = 0
    for cnt in edges:
        cnt = np.squeeze(cnt, axis=1)
        for point in cnt:
            if sq_distance(point, center)*math.sin(get_angle_HOF(point)) > prev_max_dist:
                new_extreme = point
                prev_max_dist = sq_distance(point, center)*math.sin(get_angle_HOF(point))
    return new_extreme


def prune_extrema(extrema, radius, center):
    new_extrema = []
    extrema.sort(key=lambda p: -sq_distance(center,p))
    i = 0
    while i < len(extrema):
        pt = extrema[i]
        kd = KDTree(list(extrema))
        redundant_pts = kd.query_ball_point(pt, .3*radius)
        extrema = [pt for i, pt in enumerate(extrema) if i not in redundant_pts]
        new_extrema.append(pt)
        i += 1
    return new_extrema


def get_extrema(center, path, radius=100):
    img, img_arr = get_img(path)
    edges = get_edge_points(center, img, img_arr, radius)
    max_extrema = find_max_extrema(center, img, img_arr, edges, radius)
    extrema = [max_extrema]
    for i in range(50):
        extrema.append(get_next_extrema(center, edges, extrema[-1]))
    return prune_extrema(extrema, radius, center)

if __name__ == "__main__":
    center = (2614, 654)
    files = daily_files(IMG_DIR)[8:9]
    img_path = IMG_DIR + "/" + files[0]
    print(img_path)
    img, img_arr = get_img(img_path)
    center, r = bfs_circle(img_path, center, 190)
    center = (round(center[0]), round(center[1]))
    extrema = get_extrema(center, img_path, r)
    plt.imshow(img)
    plt.plot(center[0], center[1], 'o', color="w")
    for pt in extrema:
        plt.plot(pt[0], pt[1], 'o', color="w")
    plt.show()
    