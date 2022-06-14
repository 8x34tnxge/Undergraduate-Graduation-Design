from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import spatial

from src.PartialSolver import PartialSolver
from src.Solution import Solution
from src.SpectralCluster import SpectralCluster


class DataSet(object):
    def __init__(
        self,
        uav_data,
        position_data_path,
    ):
        self.uav_data = uav_data
        self.position_data = pd.read_csv(position_data_path).loc[:100, :]
        self.max_uav_num = 75
        self.earlyPenalty = 10000
        self.latePenalty = 10000
        self.penalty = 1000

        self.dist_mat = self.calcVertexDisMatrix()
        self.position_data.to_csv("test.csv")
        self.vertex_labels = SpectralCluster().fit(
            10,
            self.position_data.loc[1:, ["XCOORD.", "YCOORD."]].to_numpy(),
            self.position_data.loc[0, ["XCOORD.", "YCOORD."]].to_numpy(),
        )

    def objFunc(self, solution: Solution):
        cost = 0
        uav_num = len(solution.uav_route)
        for routine in solution.uav_route:
            if len(routine) == 0:
                continue
            cost += self.dist_mat[0, routine[0]] + self.dist_mat[0, routine[-1]]
            for i in range(1, len(routine)):
                cost += self.dist_mat[routine[i - 1], routine[i]]

        return cost + self.penalty + uav_num
        # [ ] detailed obj function
        return cost

    def initSolution(self):
        uav_routes, uav_costs = list(), list()

        uav_route, routine_cost = list(), 0
        for label in set(self.vertex_labels):
            if label == -1:
                continue
            PartialSolver().fit(
                self.vertex_labels, label, self.dist_mat, self.feasibleFunc
            )
        solution = Solution(uav_routes, sum(uav_costs))

        return solution

    def feasibleFunc(self, solution: List):
        # [ ] do not consider it currently
        if isinstance(solution, list):
            cost = self.dist_mat[0, solution[0]] + self.dist_mat[solution[-1], 0]
            for i in range(1, len(solution)):
                cost += self.dist_mat[solution[i - 1], solution[i]]
            if cost / self.uav_data.velocity > self.uav_data.max_fly_dist:
                return False
        elif isinstance(solution, Solution):
            for routine in solution:
                cost = self.dist_mat[0, routine[0]] + self.dist_mat[routine[-1], 0]
                for i in range(1, len(routine)):
                    cost += self.dist_mat[routine[i - 1], routine[i]]
                if cost / self.uav_data.velocity > self.uav_data.max_fly_dist:
                    return False
        elif isinstance(solution, float):
            return solution / self.uav_data.velocity <= self.uav_data.max_fly_dist
        return True

    def calcVertexDisMatrix(self):
        coord = self.position_data.loc[:, ["XCOORD.", "YCOORD."]].to_numpy()
        dist_mat = spatial.distance.cdist(coord, coord, metric="euclidean")

        return dist_mat

    def calcLabelDisMatrix(self, vertexLabels):
        size = int(vertexLabels.max() + 1)
        labelDisMatrix = np.zeros([size for _ in range(2)])
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                distance = SpectralCluster.calcDistance(
                    i, j, vertexLabels, self.dist_mat
                )
                labelDisMatrix[i, j] = distance

        return labelDisMatrix


def main():
    pass


if __name__ == "__main__":
    main()
