import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class SpectralCluster(object):
    def fit(self, n_clusters, position_data, origin_vertex):
        cluster_size = np.math.floor(position_data.shape[0] / n_clusters)

        angles = np.array(
            list(map(lambda x: self.calc_angle(x, origin_vertex), position_data))
        )
        sort_index = np.argsort(angles)

        ret = np.zeros(sort_index.shape)
        for idx, elem in enumerate(sort_index):
            ret[elem] = idx // cluster_size
        ret = np.concatenate([np.array([-1]), ret])

        return ret

    @staticmethod
    def calc_angle(vector: np.ndarray, origin_vertex):
        vector = vector - origin_vertex
        vector = vector / np.linalg.norm(vector)
        angle = np.math.acos(vector[0])
        if vector[1] < 0:
            angle = 2 * np.math.pi - angle
        return angle

    @staticmethod
    def calcDistance(label_A, label_B, vertexLabels, disMatrix):
        distance = np.inf
        for index_A in np.where(np.isin(vertexLabels, label_A))[0]:
            for index_B in np.where(np.isin(vertexLabels, label_B))[0]:
                temp = disMatrix[index_A, index_B]
                if temp < distance:
                    distance = temp

        return distance
