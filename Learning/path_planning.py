import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--start', '-s', type=int, default='0', help='Day to start generating paths.')
parser.add_argument('--end', '-e', type=int, default='99', help='Day to stop generating paths.')

args = parser.parse_args()

for i in range(args.start, args.end + 1):
    origin = [0, 0]

    prune_dict, irr_coords = pickle.load(open("Coords/coords" + str(i) + ".pkl", "rb"))
    prune_coords = []
    for plant_type in prune_dict:
        for c in prune_dict[plant_type]:
            prune_coords.append(list(c[0]))
    # Set tool location as initial position
    irr_coords.insert(0, origin)
    prune_coords.insert(0, origin)
    irr_coords = np.array([list(c) for c in irr_coords])
    prune_coords = np.array(prune_coords)

    # Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
    path_distance = lambda r,c: np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])
    # Reverse the order of all elements from element i to element k in array r.
    two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))
    # From https://stackoverflow.com/questions/25585401/travelling-salesman-in-scipy
    def two_opt(cities, improvement_threshold): # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
        route = np.arange(cities.shape[0]) # Make an array of row numbers corresponding to cities.
        improvement_factor = 1 # Initialize the improvement factor.
        best_distance = path_distance(route,cities) # Calculate the distance of the initial path.
        while improvement_factor > improvement_threshold: # If the route is still improving, keep going!
            distance_to_beat = best_distance # Record the distance at the beginning of the loop.
            for swap_first in range(1,len(route)-2): # From each city except the first and last,
                for swap_last in range(swap_first+1,len(route)): # to each of the cities following,
                    new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these cities
                    new_distance = path_distance(new_route,cities) # and check the total distance with this modification.
                    if new_distance < best_distance: # If the path distance is an improvement,
                        route = new_route # make this the accepted best route
                        best_distance = new_distance # and update the distance corresponding to this route.
            improvement_factor = 1 - best_distance/distance_to_beat # Calculate how much the route has improved.
        return route # When the route is no longer improving substantially, stop searching and return the route.

    paths_dirname = "Paths/"
    if not os.path.exists(paths_dirname):    
        os.makedirs(paths_dirname)
            
    route = two_opt(prune_coords, 0.01)
    final_prune_order = np.concatenate((np.array([prune_coords[route[i]] for i in range(len(route))]), np.array([prune_coords[0]])))
    fig, ax = plt.subplots()
    ax.set_xlim([0, 150])
    ax.set_ylim([0, 150])
    plt.gca().invert_yaxis()
    plt.plot(final_prune_order[:,1],final_prune_order[:,0])
    plt.savefig(paths_dirname + 'prune_order_' + str(i) + '.png', bbox_inches='tight', pad_inches=0.02)
    plt.close()
    # Print the route as row numbers and the total distance travelled by the path.
    print("Route: " + str(route) + "\n\nDistance: " + str(path_distance(route, prune_coords)))

    route = two_opt(irr_coords, 0.01)
    final_irr_order = np.concatenate((np.array([irr_coords[route[i]] for i in range(len(route))]),np.array([irr_coords[0]])))
    fig, ax = plt.subplots()
    ax.set_xlim([0, 150])
    ax.set_ylim([0, 150])
    plt.gca().invert_yaxis()
    plt.plot(final_irr_order[:,1], final_irr_order[:,0])
    plt.savefig(paths_dirname + 'irr_order_' + str(i) + '.png', bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print("Route: " + str(route) + "\n\nDistance: " + str(path_distance(route, irr_coords)))
    
    # Save the routes to a pickle file.
    pickle.dump({'prune_order': final_prune_order, 'irr_order': final_irr_order}, open(paths_dirname + "traj_day_" + str(i) + ".pkl", "wb"))