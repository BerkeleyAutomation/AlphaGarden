#!/usr/bin/env python
import os
import argparse
import moviepy.video.io.ImageSequenceClip

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, default='./', help='Location of saved images of garden run.')
args = parser.parse_args()

image_folder = args.path
video_name = 'video.mov'

images = [image_folder + '/' + img for img in os.listdir(image_folder) if img.endswith(".png")]
images = sorted(images, key=lambda x: int(x.split('/')[-1].split('_')[0]))

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=10)
clip.write_videofile('timelapse.mp4')