import os
import argparse
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
matplotlib.use('Qt5Agg')


def create_animation(args):
    load_path = args.load_path

    fig = plt.figure(figsize=(16, 9))
    list_dicts = []

    index = 0
    for file in sorted(os.scandir(load_path), key=lambda x: x.name):
        if file.path.endswith(".p") and file.is_file():
            index += 1
            saved_dict = pickle.load(open(file, 'rb'))
            list_dicts.append(saved_dict)
            if index >= args.r * args.c:
                break

    x_dim = list_dicts[0]['x_dim']
    y_dim = list_dicts[0]['y_dim']

    for i in range(len(list_dicts)):
        ax = fig.add_subplot(args.r, args.c, i + 1)
        ax.set_xlim((0, x_dim))
        ax.set_ylim((0, y_dim))
        ax.set_aspect('equal')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_facecolor('black')

    frames = []
    for i in range(len(list_dicts[0]['plots'])):
        shape_plots = []
        for j in range(len(list_dicts)):
            ax = plt.subplot(args.r, args.c, j + 1)
            for shape in list_dicts[j]['plots'][i]:
                shape_plots.append(ax.add_artist(shape))
        frames.append(shape_plots)

    # For the last frame to be shown longer in the video
    for i in range(19):
        frames.append(shape_plots)

    growth_animation = animation.ArtistAnimation(fig, frames, interval=args.interval, blit=True, repeat=True,
                                                 repeat_delay=1000)
    fig.tight_layout(pad=0.1)

    if args.show:
        plt.show()

    growth_animation.save(args.save_path, dpi=300, savefig_kwargs={'facecolor': 'black'})


def get_parsed_args():
    parser = argparse.ArgumentParser(description='Create an animation from saved frames')
    parser.add_argument('--load_path', type=str, default='./saved_plots',
                        help='Local path to folder containing pickle files of plots creating the animation.')
    parser.add_argument('--show', dest='show', action='store_true', help='Whether to show the animation before saving')
    parser.add_argument('--interval', type=int, default=50, help='Delay in ms between frames in animation.')
    parser.add_argument('--save_path', type=str, default='simulation.mp4',
                        help='Local path to save animation to.')
    parser.add_argument('--r', type=int, default=1, help='Amount of rows of plots in visualization')
    parser.add_argument('--c', type=int, default=1, help='Amount of columns of plots in the visualization')
    return parser.parse_args()


args = get_parsed_args()
create_animation(args)
