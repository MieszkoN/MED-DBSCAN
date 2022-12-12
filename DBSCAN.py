import numpy as np
import math

UNCLASSIFIED = False
NOISE = None

def calculate_distance(p,q):
	return math.sqrt(np.power(p-q,2).sum())

def check_epsilon(p,q,eps):
	return calculate_distance(p,q) < eps

def range_query(db_values, point_q, eps):
    n_points = db_values.shape[1]
    neighbours = []
    for i in range(0, n_points):
        if check_epsilon(db_values[:,point_q], db_values[:,i], eps):
            neighbours.append(i)
    return neighbours
        
def dbscan(db_values, eps, min_points):
    cluster_id = 1
    n_points = db_values.shape[1]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        if classifications[point_id] != UNCLASSIFIED:
            continue

        neighbours = range_query(db_values, point_id, eps)
        if len(neighbours) < min_points:
            classifications[point_id] = NOISE
            continue

        cluster_id = cluster_id + 1

        classifications[point_id] = cluster_id
        for seed_id in neighbours:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = range_query(db_values, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]

    return classifications
