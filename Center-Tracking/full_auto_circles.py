from matplotlib.pyplot import show
from plant_to_circle import *
from geometry_utils import *
from center_constants import *
# from centers_test import *
from full_auto_utils import *
from run import *
import numpy as np
import pickle as pkl
import traceback


############################
######### Private ##########
############################


def label_circles_BFS(path, show_res=False):
    print("BFS Fit for: "+path)
    priors = get_recent_priors()
    new_circles = {plant_type: [] for plant_type in priors.keys()}
    # Iterate over each plant type
    for plant_type in priors.keys():
        old_circles = priors[plant_type]
        # Get radius models
        rad_models = (get_model_coeff("min", plant_type),
                      get_model_coeff("max", plant_type))
        for circle in old_circles:
            # circle = {"circle":circle, "days_post_germ": 10}
            new_c = {}
            center = circle["circle"][0]
            center = (round(center[0]), round(center[1]))
            prev_rad = circle["circle"][1]
            day = circle["days_post_germ"]+1
            min_rad, max_rad = get_radius_range(day, prev_rad, rad_models)
            try:
                c, max_p = bfs_circle(path, center, max_rad, min_rad)
                r = distance(c, max_p)
                # Check for Nan radii
                if r == r:
                    # print(c, r)
                    "continue"
                else:
                    r = 0
            except ZeroDivisionError:
                traceback.print_exc() 
                #TODO ADD WILTING LOGIC
                r, c, max_p = prev_rad, center, circle["circle"][2]
            new_c["circle"], new_c["days_post_germ"] = (c, r, max_p), day
            computed_type = COLORS_TO_TYPES[find_color(c, get_img(path)[1])[0]]
            new_circles[computed_type if computed_type in new_circles.keys() else "radiccio"].append(new_c)

    date = path[path.find("-2")+1:path.find("-2")+7]
    save_priors(new_circles, date)
    circles_dict = save_circles(new_circles, date)
    if show_res:
        show_circs = {key:[] for key in circles_dict.keys()}
        for key in new_circles.keys():
            for c in new_circles[key]:
                circle = c["circle"]
                show_circs[key].append((circle[0], circle[1]))
            show_circs[key] = merge_circles(show_circs[key])
        draw_circles(path, show_circs, True)
    return new_circles, circles_dict


def label_circles_contours(path, show_res=False):
    print("Contour Fit for: " + path)
    def get_circles_from_prior(priors):
        circles = {}
        for key in priors.keys():
            circles[key] = [priors[key][i]["circle"]
                            for i in range(len(priors[key]))]
        return circles
    priors = get_circles_from_prior(get_recent_priors())
    mapping = contour_fit_circles(path, priors)
    if show_res:
        draw_circles(path, mapping, True)
    return mapping

############################
######### Public ###########
############################

def process_image(path: str, save_circles: bool = False, crop: bool = False) -> dict:
    '''
    @param path: string representing path of the uncropped image
    @param save_circles: optionally saves circles to center_constants.py/CIRCLE_PATH
    @return dictionary of circles formatted like: 
        {
            "arugula": [
                ((center_x, center_y), radius (in cm)), 
                ...
            ], 
            ...
        }
    
    Takes uncropped image at path and runs segmentation pipeline on it. 
    First the image is cropped according to parameters using on Sept 2020 garden, 
    then the segmentation mask is extract and post-processed. Then, BFS is run with priors, 
    which are the most recent prior stored in center_constants.py/PRIOR_PATH'''
    if crop:
        path = crop_img(path)
    id_ = path[path.find(IMAGE_NAME_PREFIX):path.find(".jpg")]
    print("Extracting Mask: "+path)
    mask_path = get_img_seg_mask(id_)
    # mask_path = "./post_process/"+id_+".png"
    print("Labelling circles: "+ mask_path)
    return label_circles_BFS(mask_path, True)[1]

    

if __name__ == "__main__":
    for f in daily_files("./farmbotsony"):
        process_image("farmbotsony/" + f, True, True)