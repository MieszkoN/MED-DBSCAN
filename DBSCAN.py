import numpy as np
import math

UNCLASSIFIED = False
NOISE = -1

def minkowski3(p,q):
    return math.pow(np.power(p-q,3).sum(),1/3)

def manhattan(p,q):
    return (np.abs(p-q).sum())

def euclidean(p,q):
    return math.sqrt(np.power(p-q,2).sum())

def check_epsilon(p,q,eps, dist_func):
	return dist_func(p,q) < eps

def range_query(db_values, point_q, eps, dist_func):
    n_points = db_values.shape[0]
    neighbours = []
    for i in range(0, n_points):
        if check_epsilon(db_values[point_q,:], db_values[i,:], eps, dist_func):
            neighbours.append(i)
    return neighbours
        

def mark_core_point(point_id, cluster_id, neighbours, seeds, labels):
    labels[point_id] = cluster_id

    for n in neighbours:
        if labels[n] is UNCLASSIFIED: # n was not classified
            labels[n] = cluster_id
            seeds.append(n)
        if labels[n] is NOISE: # n is not core, but add to group
            labels[n] = cluster_id
        else:
            continue # n is already assigned to a group
    
    return labels, seeds

def dbscan(db_values, eps, min_points, dist_func: str = 'euclidean'):

    func = euclidean

    if dist_func == 'euclidean':
        func = euclidean
    elif dist_func == 'manhattan':
        func = manhattan
    elif dist_func == 'minkowski':
        func = minkowski3

    seeds = []
    real_seeds = {}
    cluster_id = 0
    n_points = db_values.shape[0]
    labels = [UNCLASSIFIED] * n_points

    for point_id in range(0,n_points):

        # get next point if this point was already considered 
        if labels[point_id] != UNCLASSIFIED:
            continue
        
        neighbours = range_query(db_values, point_id, eps, dist_func=func)
            
        if len(neighbours) < min_points:
            labels[point_id] = NOISE
            continue
            
        # start next cluster
        cluster_id = cluster_id + 1
        # add point to final seeds
        real_seeds[cluster_id] = point_id
        # deal with core point and its neighbours
        labels, seeds = mark_core_point(point_id, cluster_id, neighbours, seeds, labels)

        while len(seeds) > 0:
            # start with first seed    
            s = seeds[0]
            # check neighbours for each seed
            s_neighbours = range_query(db_values, s, eps, dist_func=func)
            if len(s_neighbours) >= min_points: # check if s is a core point
                # deal with core point and its neighbours
                mark_core_point(point_id=s, cluster_id=cluster_id, neighbours=s_neighbours, seeds=seeds, labels=labels)
            seeds = seeds[1:]
            
    return labels, real_seeds