from full_auto_utils import crop_img, get_recent_priors
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import KDTree
from center_constants import *
from geometry_utils import *
from centers_test import *
from plant_to_circle import *
from tqdm import tqdm
import sys

'''
How to run this script: 
python3 linearity.py {PRIOR_PATH} {MASK_PATH}

'''

def get_plant_type(center, img_arr):
    center = (round(center[0]), round(center[1]))
    rgb_center, first_color_pixel = find_color(center, img_arr)
    return COLORS_TO_TYPES[rgb_center]

def get_nearby_contours(contours, center, inverted, radius):
    #TODO: SPEEDUP
    # nearby = []
    # for cnt in contours:
    #     # hull = cv2.convexHull(cnt)
    #     # pair = approximate_circle_contour(inverted, hull)
    #     pair = np.average(cnt, axis=0)
    #     # print(pair)
    #     if distance(center, pair[0]) < radius:
    #         nearby.append(cnt)
    # return nearby
    return list(filter(lambda cnt: distance(center, np.average(cnt[0], axis=0)) < radius, contours))


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
    for _ in range(50):
        extrema.append(get_next_extrema(center, edges, extrema[-1]))
    return prune_extrema(extrema, radius, center)

def get_leaf_center(extrema, center):
    return ((extrema[0]*.5 + center[0]*.5), (extrema[1]*.5 + center[1]*.5))


def get_max_leaf_centers(prior, mask_path, only_right=False):
    ''' Returns the center of the leaf the largest leaf of each plant based on get_leaf_center algorithm.
    Params
        :prior: Dictonary of each plant type containing prior locations and radii of each plant
        :mask_path: Path of the mask being used to 
    Returns
        :extreme_pts: The leaf centers according to the algorithm
    '''
    extreme_pts = []
    for key in tqdm(prior):
        for p in prior[key]:
            center, r = p["circle"][0:2]
            if only_right and center[0] < 1630: #Value to be tuned to dictate each half of garden
                continue
            extrema = get_extrema(center, mask_path, 1.2*r)
            extreme_pts.append((center, get_leaf_center(max(extrema, key=lambda p: distance(center, p)), center)))
    return extreme_pts


if __name__ == "__main__":
    # Get priors and image from sysargs
    prior = get_recent_priors(str(sys.argv[1]))
    mask_path = str(sys.argv[2])
    print(mask_path)
    mask, _ = get_img(mask_path)

    # This gets the actual overhead image
    # real_path = "input/new_garden/snc-21052608141500.jpg"
    plt.imshow(mask)

    # UNCOMMENT THIS TO PLOT ALL EXTREMA
    # leaf_centers = []
    # for key in tqdm(prior):
    #     for circle in prior[key]:
    #         center, r = circle["circle"][:2]
    #         if center[0] < 1600:
    #             continue
    #         leaf_centers.append(center)
    #         leaf_centers.extend(get_extrema(center, mask_path, 2*r))
    # for pt in leaf_centers:
    #     plt.plot(pt[0], pt[1], '.', color="w", markersize=4)
    # plt.savefig("organs_2.jpg", bbox_inches = 'tight', pad_inches = 0, dpi=500)

    leaf_centers = get_max_leaf_centers(prior, mask_path, True)
    for pt in leaf_centers:
        plt.plot(pt[0], pt[1], '.', color="w", markersize=4)
    # For debugging 
    print(leaf_centers)
    if "prune_points" not in os.listdir("."):
        os.mkdir("prune_points") 
    circles_file = sys.argv[1][:-1][sys.argv[1][:-1].find("iors/") + 4:]
    save_centers("prune_points/"+circles_file+"txt", leaf_centers)
    plt.savefig("leaf_center.jpg", bbox_inches = 'tight', pad_inches = 0, dpi=500)
    
