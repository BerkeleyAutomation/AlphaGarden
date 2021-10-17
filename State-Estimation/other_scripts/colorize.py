from full_auto_utils import *
import pickle as pkl

"""
Takes an overhead image and Center Tracking prior to create an overlaid image
using colored circles.
"""

DATE = "210525" #YYMMDD
overhead_image = "../cropped/snc-21052520141400.jpg"
circles = get_recent_priors("../priors/priors" + DATE + ".p")
centers, radii, colors = [], [], []
for color in COLORS_TO_TYPES.keys():
    type = COLORS_TO_TYPES[color]
    if type not in circles:
        continue
    c = circles[type]
    cur_cen, cur_rad = [], []
    for circ in c:
        c, rad, _ = circ["circle"]
        cur_cen.append(c)
        cur_rad.append(rad)
    centers.append(cur_cen)
    radii.append(cur_rad)
    colors.append(color)
print(colors)
# print(centers, radii)
draw_circle_sets(overhead_image, centers, radii, colors)
