from PIL import Image, ImageDraw, ImageFont
from plant_presets import *
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, default='./', help='Location of seed placement file. ex.: data/f_seed100_overlap0.1_trial0')
args = parser.parse_args()

CANVAS_SIZE = 5

def draw_image(labels, points):
    im = Image.new("RGB", (150*CANVAS_SIZE,150*CANVAS_SIZE), (255,255,255))
    dr = ImageDraw.Draw(im)
    for i in range(len(points)):
        [x,y] = points[i]
        l = labels[i]
        type = PLANTS[l]
        r = PLANT_SIZE[type]
        color = COLORS[l]
        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        dr.ellipse(((x-r)*CANVAS_SIZE,(y-r)*CANVAS_SIZE,(x+r)*CANVAS_SIZE,(y+r)*CANVAS_SIZE), fill=(color[0] + round((255-color[0])*2/4), color[1]+round((255-color[1])*2/4), color[2]+round((255-color[2])*2/4)), outline=color, width=1)
        dr.ellipse((x*CANVAS_SIZE-3,y*CANVAS_SIZE-3,x*CANVAS_SIZE+3,y*CANVAS_SIZE+3), color)
    return im

with open(args.path, "rb") as f:
    [labels, points] = pickle.load(f)
    # for i in range(len(labels)):
    #     print(labels[i], points[i]) 

overlap_im = draw_image(labels, points)
overlap_im.save("orig_seed_placement.png")