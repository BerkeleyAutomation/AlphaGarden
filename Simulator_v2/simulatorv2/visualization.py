from logger import Event
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def plot_garden(garden):
    """
    Displays the current state of the garden, including:
    - All plants and their sunlight points
    - Soil water levels
    """
    fig, ax = _setup_plot(garden)
    _add_plots(garden, ax)
    plt.show()

def plot_data(garden, num_timesteps):
    """
    Displays graphs of all events that were recorded during timesteps of the simulation
    (e.g. plant radius and height over time, water absorbed over time)
    """
    for event_type in Event:
        plt.figure()
        for plant in garden.plants.values():
            plt.plot(range(num_timesteps), garden.logger.get_data(event_type, plant.id), color=plant.color)
        # plt.plot(range(NUM_TIMESTEPS), garden.logger.get_data(event_type, "Control"), color='0.5', linestyle='--')
        plt.title(event_type.value)
    plt.show()


##################################################
##         HELPER FUNCTIONS (private)           ##
##################################################

def setup_animation(garden):
    """
    NO NEED TO CALL THIS DIRECTLY! This is used by the Garden class to support animations.
    If you want to animate a simulator run, pass in `animate=True` when initializing your Garden,
    and call show_animation() at the end.

    --------
    Performs necessary setup to allow animating the garden's history.

    Returns two functions:
    (1) anim_step should be called at each step of the garden; it records all information
        that will appear in the animation, e.g. plant locations/radii and water levels
    (2) anim_show should be called ONCE; it will open a window and show an animation of all
        the steps recorded
    """
    frames = []
    fig, ax = _setup_plot(garden)

    def anim_step():
        frames.append(_add_plots(garden, ax))

    def anim_show():
        growth_animation = animation.ArtistAnimation(fig, frames, interval=300, blit=True, repeat_delay=1000)
        plt.show()
        growth_animation.save('simulation.mp4')

    return anim_step, anim_show

def _setup_plot(garden):
    """
    Helper function to set up (but NOT show) the matplotlib grid that will be used
    to visualize the garden.
    """
    fig, ax = plt.subplots()
    plt.xlim((0, garden.N * garden.step))
    plt.ylim((0, garden.M * garden.step))
    ax.set_aspect('equal')

    major_ticks = np.arange(0, garden.N * garden.step + 1, garden.N // 5)
    minor_ticks = np.arange(0, garden.N * garden.step + 1, garden.step)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    return fig, ax

def _add_plots(garden, ax):
    """
    Helper function to record the current state of the garden (plants, sunlight points and water levels)
    on the given set of axes.

    Returns a list of all shapes that were added to the plot.
    """
    shapes = []
    for plant in sorted(garden.plants.values(), key=lambda plant: plant.height, reverse=True):
        circle = plt.Circle((plant.row, plant.col) * garden.step, plant.radius, color=plant.color)
        circleplot = ax.add_artist(circle)
        shapes.append(circleplot)
    for coord, water_amt in garden.get_water_amounts():
        circle = plt.Circle(coord * garden.step, water_amt / 100, color='b', alpha=0.3)
        circleplot = ax.add_artist(circle)
        shapes.append(circleplot)
    for grid_pt, coord in garden.enumerate_grid(coords=True):
        if grid_pt['nearby']:
            circle = plt.Circle(coord * garden.step, 0.2, color='c', alpha=0.3)
            circleplot = ax.add_artist(circle)
            shapes.append(circleplot)

    return shapes
