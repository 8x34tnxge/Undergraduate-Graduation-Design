import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Solution import Solution
from VNS import VariableNeighborhoodSearch


class PartialSolver(object):
    def fit(self, vertexLabels, label, disMatrix, is_feasible):
        labelIdx = np.where(np.isin(vertexLabels, label))[0]
        solution = self.generateRoute(labelIdx, disMatrix, is_feasible)

        return solution

    def generateRoute(self, label_idx, disMatrix, is_feasible):
        vnsModel = VariableNeighborhoodSearch(label_idx, disMatrix, is_feasible)
        ret = vnsModel.fit()

        return ret


def main():
    pass


if __name__ == "__main__":
    main()
