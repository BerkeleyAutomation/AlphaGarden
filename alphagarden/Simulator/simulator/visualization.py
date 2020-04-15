'''
from alphagarden.Simulator.simulator.logger import Event
from alphagarden.Simulator.simulator.plant_stage import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
matplotlib.use('Qt5Agg')  # Only keep when running on Mac


def plot_garden(garden):
    """
    Displays the current state of the garden, including:
    - All plants and their sunlight points
    - Soil water levels
    """
    fig, ax = _setup_plot(garden)
    _add_frames(garden, ax, reverse=True)
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


def plot_calibration_curve(plant, garden_width, garden_height, num_timesteps, min_water=0, max_water=1, step_size=0.1):
    """
    Plots a calibration curve for the given plant type.
    The calibration curve shows the attainable final canopy cover as a function of constant watering amounts,
    to provide a baseline measurement. It assumes the plant has no competition and gets full sunlight.

    x-axis: constant water provided at each timestep
    y-axis: final canopy cover as a result of watering this single unoccluded plant for `num_timesteps` steps

    The plot will display for values between `min_water` and `max_water`, in increments of `step_size`.
    """
    water_amt = min_water
    water_values = []
    cc_values = []
    while water_amt <= max_water:
        plant.start_from_beginning()
        for _ in range(num_timesteps):
            plant.num_sunlight_points = plant.num_grid_points
            plant.water_amt = water_amt
            upward, outward = plant.amount_to_grow()
            plant.height += upward
            plant.radius += outward
            plant.reset()
            plant.num_grid_points = np.pi / 4 * (plant.radius * 2 + 1) ** 2
        water_values.append(water_amt)
        print(plant.radius, plant.num_grid_points)
        cc_values.append(plant.num_grid_points / (garden_width * garden_height))
        water_amt += step_size

    plt.figure()
    plt.title("Constant water amount vs. final canopy cover")
    plt.plot(water_values, cc_values)
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
        frames.append(_add_frames(garden, ax))

    def anim_show():
        growth_animation = animation.ArtistAnimation(fig, frames, interval=25, blit=True, repeat=True,
                                                     repeat_delay=1000)
        fig.set_size_inches(10, 5, True)
        plt.show()
        growth_animation.save('simulation.mp4', dpi=300)

    return anim_step, anim_show


"""
    NO NEED TO CALL THIS DIRECTLY! This is used by the Garden class to support saving data for animations.
    
    --------
    Performs necessary setup to allow saving data for animating the garden's history.

    Returns three functions:
    (1) save_step should be called at each step of the garden; it records all information
        that will appear in the animation, e.g. plant locations/radii
    (2) save_final_step should be called ONCE after the simulation is complete; it will save the final state of the
        garden without any indicators of pruning action
    (3) get_plots returns all of the saved information, for plotting when creating the animation
    """

def setup_saving(garden):
    plots = []

    def save_step():
        plots.append(_add_plots(garden))

    def save_final_step():
        plots.append(_add_plots(garden, final=True))

    def get_plots():
        return plots

    return save_step, save_final_step, get_plots


def _setup_plot(garden):
    """
    Helper function to set up (but NOT show) the matplotlib grid that will be used
    to visualize the garden.
    """
    fig, ax = plt.subplots()
    plt.xlim((0, (garden.N - 1) * garden.step))
    plt.ylim((0, (garden.M - 1) * garden.step))
    ax.set_aspect('equal')

    # ax.set_xticks(np.arange(0, garden.N * garden.step + 1, max(garden.N // 5, 1)))
    # ax.set_xticks(np.arange(0, garden.N * garden.step + 1, garden.step), minor=True)
    # ax.set_yticks(np.arange(0, garden.M * garden.step + 1, max(garden.N // 5, 1)))
    # ax.set_yticks(np.arange(0, garden.M * garden.step + 1, garden.step), minor=True)
    # ax.grid(which='minor', alpha=0.2)
    # ax.grid(which='major', alpha=0.5)

    # TEMPORARY, FOR DEMO ONLY
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])

    return fig, ax


def _add_frames(garden, ax, reverse=False):
    """
    Helper function to record the current state of the garden (plants, sunlight points and water levels)
    on the given set of axes.

    Returns a list of all shapes that were added to the plot.
    """
    shapes = []

    # TODO: This only works when step = 1
    # Heatmap of soil water levels
    # c = plt.imshow(garden.grid['water'].T, cmap='Blues', origin='lower', alpha=0.5)
    # cp = ax.add_artist(c)
    # shapes.append(cp)

    # Plants
    for plant in sorted([plant for plant_type in garden.plants for plant in plant_type.values()],
                        key=lambda x: x.height, reverse=reverse):
        if plant.pruned:
            shape = plt.Rectangle((plant.row * garden.step - plant.radius,
                                  plant.col * garden.step - plant.radius), plant.radius * 2, plant.radius * 2,
                                  fc='red', ec='red')
        else:
            shape = plt.Circle((plant.row, plant.col) * garden.step, plant.radius, color=plant.color)
        shape_plot = ax.add_artist(shape)
        shapes.append(shape_plot)

    # Sunlight points
    # for grid_pt, coord in garden.enumerate_grid(coords=True):
    #     if grid_pt['nearby']:
    #         circle = plt.Circle(coord * garden.step, 0.2, color='c', alpha=0.3)
    #         circleplot = ax.add_artist(circle)
    #         shapes.append(circleplot)

    # Irrigation points
    # for (row, col), amount in garden.irrigation_points.items():
    #     square = plt.Rectangle(((row - 0.5) * garden.step, (col - 0.5) * garden.step), garden.step, garden.step,
    #                            fc='red', ec='red')
    #     squareplot = ax.add_artist(square)
    #     shapes.append(squareplot)

    return shapes


def _add_plots(garden, reverse=False, final=False):
    shapes = []

    for plant in sorted([plant for plant_type in garden.plants for plant in plant_type.values()],
                        key=lambda x: x.height, reverse=reverse):
        if not final and plant.pruned:
            shape = plt.Rectangle((plant.row * garden.step - plant.radius,
                                  plant.col * garden.step - plant.radius), plant.radius * 2, plant.radius * 2,
                                  fc='red', ec='red')
        else:
            shape = plt.Circle((plant.row, plant.col) * garden.step, plant.radius, color=plant.color)
        shapes.append(shape)

    return shapes
'''
