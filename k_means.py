import numpy as np
import random

def calculate_euclidean_distance(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))

class KMeans:
    def __init__(self, clusters_number, max_iterations):
        self.clusters_number = clusters_number
        self.max_iterations = max_iterations
        
    def fit(self, df):
        self.centroids = [random.choice(df)]
        for _ in range(self.clusters_number - 1):
            distance_list = np.sum([calculate_euclidean_distance(centroid, df) for centroid in self.centroids], axis=0)
            distance_list /= np.sum(distance_list)
            new_centroid_id, = np.random.choice(range(len(df)), size = 1, p = distance_list)
            self.centroids += [df[new_centroid_id]]

        iteration = 0
        previous_centroids = None
        while np.not_equal(self.centroids, previous_centroids).any() and iteration < self.max_iterations:
            sorted_points = [[] for _ in range(self.clusters_number)]
            for x in df:
                distance_list = calculate_euclidean_distance(x, self.centroids)
                centroid_idx = np.argmin(distance_list)
                sorted_points[centroid_idx].append(x)
            previous_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  
                    self.centroids[i] = previous_centroids[i]
            iteration += 1
            
    def group_centroids(self, X):
        centroids = []
        centroid_ids_list = []
        for x in X:
            distance_list = calculate_euclidean_distance(x, self.centroids)
            centroid_id = np.argmin(list(distance_list))
            centroids.append(self.centroids[centroid_id])
            centroid_ids_list.append(centroid_id)
        return np.array(centroids), np.array(centroid_ids_list)