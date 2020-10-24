import pickle
from PIL import Image, ImageDraw, ImageFont

REAL_GARDEN_WIDTH = 118.11 / 2 # in inches
ratio = 100 / REAL_GARDEN_WIDTH

NUM_PIXELS = 100

def get_r_max(v):
    r_max = (v / 2) * ratio
    return r_max

SEED_SPACING = {
 'borage': 15,
 'mizuna': 10,
 'mustard': 5,
 'sorrel': 12,
 'cilantro': 6,
 'radicchio': 9,
 'kale': 15,
 'green_lettuce': 12,
 'red_lettuce': 10,
 'arugula': 12,
 'swiss_chard': 14,
 'turnip': 9,
 'mint': 12
}

PLANT_SIZE = {}
for key in SEED_SPACING:
    PLANT_SIZE[key] = get_r_max(SEED_SPACING[key])

PLANTS = ['borage', 'sorrel', 'cilantro', 'radicchio', 'kale', 'green_lettuce', 'red_lettuce', 'arugula', 'swiss_chard', 'turnip']

# SAME_RELATIONSHIP_VALUE
SRV = 1.0

PLANTS_RELATION = {
        "borage":       {"borage": SRV, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": 0.0, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "arugula": 0.0, "swiss_chard": 0.0, "turnip": 1.0},
        "sorrel":       {"borage": 0.0, "sorrel": SRV,  "cilantro": 0.0, "radicchio": 0.0, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "arugula": 0.0, "swiss_chard": 0.0, "turnip": 0.0},
        "cilantro":     {"borage": 0.0, "sorrel": 0.0,  "cilantro": SRV, "radicchio": 0.0, "kale": 0.0, "green_lettuce": 1.0, "red_lettuce": 1.0, "arugula": 0.0, "swiss_chard": 0.0, "turnip": 0.0},
        "radicchio":    {"borage": 0.0, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": SRV, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "arugula": 0.0, "swiss_chard": 0.0, "turnip": 0.0},
        "kale":         {"borage": 0.0, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": 0.0, "kale": SRV, "green_lettuce": 0.0, "red_lettuce": 0.0, "arugula": 0.0, "swiss_chard": 0.0, "turnip": 0.0},
        "green_lettuce":{"borage": 0.0, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": 1.0, "kale": 0.0, "green_lettuce": SRV, "red_lettuce": 0.0, "arugula":-1.0, "swiss_chard": 0.0, "turnip": 0.0},
        "red_lettuce":  {"borage": 0.0, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": 1.0, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": SRV, "arugula":-1.0, "swiss_chard": 0.0, "turnip": 0.0},
        "arugula":      {"borage": 0.0, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": 0.0, "kale": 0.0, "green_lettuce":-1.0, "red_lettuce":-1.0, "arugula": SRV, "swiss_chard": 0.0, "turnip": 0.0},
        "swiss_chard":  {"borage": 0.0, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": 0.0, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "arugula": 0.0, "swiss_chard": SRV, "turnip": 1.0},
        "turnip":       {"borage": 0.0, "sorrel": 0.0,  "cilantro": 0.0, "radicchio": 0.0, "kale": 0.0, "green_lettuce": 0.0, "red_lettuce": 0.0, "arugula": 0.0, "swiss_chard": 1.0, "turnip": SRV}
}

COLORS = [(255,128,0),(200,200,0),(0,255,0),(0,255,255),(0,128,255),(0,0,255),(128,0,255),(255,0,255),(128,128,128),(204,229,255),(0,0,0),(255,0,0)]
# COLORS = [(9 / 255, 77 / 255, 10 / 255), (167 / 255, 247 / 255, 77 / 255), (101 / 255, 179 / 255, 53 / 255), (147 / 255, 199 / 255, 109 / 255), (117 / 255, 158 / 255, 81 / 255), (142 / 255, 199 / 255, 52 / 255), (117 / 255, 128 / 255, 81 / 255), (58 / 255, 167 / 255, 100 / 255), (58 / 255, 137 / 255, 100 / 255), (0, 230 / 255, 0)]
RATIO = 5
from PIL import Image, ImageDraw, ImageFont
# x, y = 100, 200
# im = Image.new("RGB", (100*RATIO*5+y+100, 100*RATIO+x+100), (200,200,200))
# dr = ImageDraw.Draw(im)
# fnt = ImageFont.truetype("../InitialPlantPlacementHelper/Arial.ttf", 20)
# for i,p1 in enumerate(PLANTS):
#     for j,p2 in enumerate(PLANTS):
#         s = PLANTS_RELATION[p1][p2]
#         if s == 0:
#             dr.rectangle((y+j*10*RATIO*5, x+i*10*RATIO, y+(j+1)*10*RATIO*5, x+(i+1)*10*RATIO), fill=(255,255,255), outline=(0,0,0), width=1)
#         elif s > 0:
#             dr.rectangle((y+j*10*RATIO*5, x+i*10*RATIO, y+(j+1)*10*RATIO*5, x+(i+1)*10*RATIO), fill=(0,200,0), outline=(0,0,0), width=1)
#         else:
#             dr.rectangle((y+j*10*RATIO*5, x+i*10*RATIO, y+(j+1)*10*RATIO*5, x+(i+1)*10*RATIO), fill=(200,0,0), outline=(0,0,0), width=1)
#
# im.save("two.png")


# PLANTS = ['borage', 'mizuna', 'sorrel', 'cilantro', 'radicchio', 'kale', 'green_lettuce', 'red_lettuce', 'arugula', 'swiss_chard', 'turnip']
# fnt = ImageFont.truetype("../InitialPlantPlacementHelper/Arial.ttf", 20)
#
# im = Image.new("RGB", (1700, 50), (255,255,255))
# dr = ImageDraw.Draw(im)
# for i in range(len(PLANTS)):
#     dr.text( (i * 140 + 15, 12),PLANTS[i], (0,0,0), font=fnt)
#     dr.ellipse((i*140, 10, i*140+10, 20), COLORS[i])
#
# im.save("dddd2.png")


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
        #dr.ellipse(((x-r)*CANVAS_SIZE,(y-r)*CANVAS_SIZE,(x+r)*CANVAS_SIZE,(y+r)*CANVAS_SIZE), fill=color, outline=color, width=1)
        # dr.ellipse((x*CANVAS_SIZE-3,y*CANVAS_SIZE-3,x*CANVAS_SIZE+3,y*CANVAS_SIZE+3), (40,40,40))

    return im

# with open('/data/scaled_orig_placement', "rb") as f:
#     [labels, points] = pickle.load(f)
#
# overlap_im = draw_image(labels, points)
# overlap_im.save("orig_seed_placement.png")
