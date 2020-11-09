import imageio
from PIL import Image
import numpy as np

filename = '/Users/sebastianoehme/Downloads/time_lapse.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')

width, height = (3478, 1630)
left = int(width/2)

i = 1
j = 1
path = '/Users/sebastianoehme/AlphaGarden/Learning/images/full/Pillow/20201017-220222/'
suf = '_cc.png'
joined_images = []

for garden_frame in vid.iter_data():
    #pill_image = Image.new("RGBA", (1879, 1849)) # left
    pill_image = Image.new("RGBA", (1849, 1849)) # right

    #sim_frame = imageio.imread(path+str(j)+suf) # for side by side

    sim_frame = Image.open(path+str(j)+suf)
    cropped_garden_frame = Image.fromarray(garden_frame[:, left:, :],'RGB')
    #imgs_comb = np.hstack((sim_frame, cropped_garden_frame)) # for side by side
    #pill_image = Image.fromarray(imgs_comb, 'RGB') # for side by side

    #pill_image.paste(cropped_garden_frame, (0, 152), cropped_garden_frame.convert('RGBA')) # left
    #pill_image.paste(sim_frame, (29, 0), sim_frame) # left
    pill_image.paste(cropped_garden_frame, (86, 140), cropped_garden_frame.convert('RGBA')) # right
    pill_image.paste(sim_frame, (0, 0), sim_frame)  # right

    pill_image.thumbnail((900, 900))
    joined_images.append(pill_image)
    i += 1
    if i % 2 == 0:
        j += 1

joined_images[0].save('/Users/sebastianoehme/AlphaGarden/Learning/images/out_irr_2_right_fit.gif',
                      save_all=True, append_images=joined_images[1:], optimize=False, duration=200, loop=0)
