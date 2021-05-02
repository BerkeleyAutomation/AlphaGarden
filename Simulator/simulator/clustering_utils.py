import scipy.cluster.hierarchy as hcluster
from sklearn.neighbors import NearestCentroid
import os
import numpy as np
from simulator.plant_stage import GerminationStage, GrowthStage, WaitingStage, WiltingStage, DeathStage


def naive_centroid(arr):
        """ Find the centroid for a set of points

        Args:
            arr (numpy array of [[int,int], ...]): Set of points to cluster
        Returns:
            (1,2) numpy array for the centroid
        """
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return np.array((sum_x/length, sum_y/length)).reshape((1,2))

def find_point_clusters(points, thresh, name = 1):
    """ Find the Centroids for a given set of points by distance threshold

    Args:
        points: (numpy array with each element(col,row))): Set of points to cluster
        thresh (float): Max distance for clustering
    
    Returns:
        numpy array of centers for the clusters
    """
    if len(points) > 1:
        clusters = hcluster.fclusterdata(points, thresh, criterion="distance")
        unique = len(np.unique(clusters))
        if unique > 1:
            clf = NearestCentroid()
            clf.fit(points,clusters)
            return clf.centroids_
        elif unique == 1:
            return naive_centroid(points)
    elif len(points) == 1:
        return points
    return np.empty((0,2),dtype=np.int)

def cluster_all_plant_centers(plants, grow_thresh = 8, germination_thresh = 8, prune_thresh = 2.5):
    """ Find all centroids for plants and their growing and germination thresholds

    Args:
        plants: The plants array from a garden instance
        grow_thresh (float): Maximum distance for plants in the growth stage
        germination_thresh (float): Maximum distance for points in germination stage
    
    Returns:
        numpy array of optimal centers to water for the given plantset
    """
    # Split all the plants into growth and germination for the decision tree process
    all_plants = [plant for plant_type in plants for plant in plant_type.values()]
    growth_plants = [plant for plant in all_plants if isinstance(plant.current_stage(), GrowthStage) ]
    germination_plants = [plant for plant in all_plants if isinstance(plant.current_stage(), GerminationStage)]
    prune_plants = []
    centers = np.empty((0,2), dtype=np.int)
    
    # Generate clusters for growing plants
    growing_plant_coords = np.array([(plant.row,plant.col) for plant in growth_plants])
    centers =  np.concatenate((centers,
            find_point_clusters(growing_plant_coords, grow_thresh, name = "grow"))).astype(np.int)
    
    # Find all germination points that fall within the growth clusters and remove them
    remove_idx = []
    for i, p in enumerate(germination_plants):
        potential = list(map(lambda c: (p.row-c[0])*(p.row-c[0]) + (p.col-c[1])*(p.col-c[1]), centers))
        if len(potential) > 0 and np.min(potential) <= germination_thresh * germination_thresh:                
            remove_idx.append(i)
    germination_plants = np.delete(germination_plants, remove_idx, axis = 0)

    # Generate clusters for remaining germination plants
    germinating_plant_coords = np.array([(plant.row,plant.col) for plant in germination_plants])
    centers =  np.concatenate((centers,
            find_point_clusters(germinating_plant_coords, germination_thresh, name = "germ"))).astype(np.int)
    
    remove_idx = []
    for i, p in enumerate(prune_plants):
        potential = list(map(lambda c: (p.row-c[0])*(p.row-c[0]) + (p.col-c[1])*(p.col-c[1]), centers))
        if len(potential) > 0 and np.min(potential) <= prune_thresh * prune_thresh:                
            remove_idx.append(i)
    prune_plants = np.delete(prune_plants, remove_idx, axis = 0)
    pruning_plant_coords = np.array([(plant.row,plant.col) for plant in prune_plants])
    centers =  np.concatenate((centers,
            find_point_clusters(pruning_plant_coords, prune_thresh, name = "prune"))).astype(np.int)

    print(f'cl{len(centers)}')
    return centers