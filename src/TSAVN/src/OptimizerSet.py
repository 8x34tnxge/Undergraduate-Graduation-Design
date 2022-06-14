import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PartialSolver import PartialSolver
from Solution import Solution


class Optimizer(object):
    def __init__(self, dataset, prob_weight, tabuRule=None):
        self.vertexLabels = dataset.vertex_labels
        self.positionData = dataset.position_data
        self.objFunc = dataset.objFunc
        self.feasibleFunc = dataset.feasibleFunc
        self.disMatrix = dataset.dist_mat
        self.calcLabelDisMatrix = dataset.calcLabelDisMatrix
        self.tabuRule = tabuRule
        if tabuRule is not None:
            self.tabuRule.initialize(len(self.positionData))
        self.optimizerNum = 5
        self.probWeight = prob_weight
        self.probNeighborhood = np.array(
            [1 / self.optimizerNum for _ in range(self.optimizerNum)]
        )
        self.selectionCounter = np.array([0 for _ in range(self.optimizerNum)])
        self.successCounter = np.array([0 for _ in range(self.optimizerNum)])

    def optimizer(self, solution: Solution, cost):
        vertexLabels = self.vertexLabels
        optimizers = [
            self.randTransfer,
            self.closestTransfer,
            self.randSwap,
            self.closestSwap,
            self.remove2NewClusters,
        ]
        for _ in range(10):
            rnd = random.random()
            selectedIdx = -1
            for idx, prob in enumerate(self.probNeighborhood):
                if rnd < prob:
                    selectedIdx = idx
                    selectedOptimizer = optimizers[idx]
                    self.selectionCounter[idx] += 1
                    break
                rnd -= prob
            vertexLabels = selectedOptimizer(vertexLabels)
        ret = Solution([], 0)
        for label in set(vertexLabels):
            if label == -1:
                continue
            partialSolverModel = PartialSolver()
            solution = partialSolverModel.fit(
                vertexLabels, label, self.disMatrix, self.feasibleFunc
            )
            ret.uav_route.extend(solution.uav_route)
            ret.routine_cost += solution.routine_cost

        if ret.routine_cost < cost:
            self.successCounter[selectedIdx] += 1
        for idx in range(len(self.probNeighborhood)):
            self.probNeighborhood[idx] *= self.probWeight
            if self.selectionCounter[idx] == 0:
                continue
            self.probNeighborhood[idx] += (
                (1 - self.probWeight)
                * self.successCounter[idx]
                / self.selectionCounter[idx]
            )
        self.probNeighborhood /= self.probNeighborhood.sum()

        return ret, ret.routine_cost

    def randTransfer(self, vertexLabels):
        rndLabel = np.random.randint(0, vertexLabels.max() + 1, 1)
        remIndex = np.where(np.isin(vertexLabels, rndLabel, invert=True))[0]
        if remIndex.shape[0] == 1:
            return vertexLabels
        rndIndex = remIndex[np.random.randint(1, remIndex.shape[0], 1)]

        if self.tabuRule is not None:
            tabuList = self.tabuRule.getAvailableList()
            if not tabuList[rndIndex]:
                remIndex = remIndex[tabuList[remIndex]]
                if remIndex.shape[0] == 1:
                    return vertexLabels
                rndIndex = remIndex[np.random.randint(1, remIndex.shape[0], 1)]
            self.tabuRule.updateTabuList(rndIndex)

        selectedLabel = vertexLabels[rndIndex]
        vertexLabels[rndIndex] = rndLabel

        if not np.where(np.isin(vertexLabels, selectedLabel))[0].size:
            vertexLabels[np.where(vertexLabels > selectedLabel)[0]] -= 1

        return vertexLabels

    def closestTransfer(self, vertexLabels):
        labelDisMatrix = self.calcLabelDisMatrix(vertexLabels)
        rndLabel = np.random.randint(0, vertexLabels.max() + 1, 1)
        *_, closestLabel = np.where(
            np.isin(labelDisMatrix[rndLabel, :], labelDisMatrix[rndLabel, :].min())
        )
        selectedIndex = np.where(np.isin(vertexLabels, rndLabel))[0]
        rndSelectedIndex = selectedIndex[
            np.random.randint(0, selectedIndex.shape[0], 1)
        ]

        if self.tabuRule is not None:
            tabuList = self.tabuRule.getAvailableList()
            if not tabuList[rndSelectedIndex]:
                candidate = [x for x in range(1, vertexLabels.max() + 1)]
                while selectedIndex.shape[0] == 1 and not tabuList[selectedIndex[0]]:
                    candidate.remove(rndLabel)
                    if len(candidate) == 0:
                        return vertexLabels
                    idx = np.random.randint(0, len(candidate), 1)
                    rndLabel = np.array(candidate)[idx]
                    selectedIndex = np.where(np.isin(vertexLabels, rndLabel))[0]
                selectedIndex = selectedIndex[tabuList[selectedIndex]]
                rndSelectedIndex = selectedIndex[
                    np.random.randint(0, selectedIndex.shape[0], 1)
                ]
            self.tabuRule.updateTabuList(rndSelectedIndex)

        vertexLabels[rndSelectedIndex] = closestLabel
        if not np.where(np.isin(vertexLabels, rndLabel))[0].size:
            vertexLabels[np.where(vertexLabels > rndLabel)] -= 1

        return vertexLabels

    def randSwap(self, vertexLabels):
        rndLabel = np.random.randint(0, vertexLabels.max() + 1, 1)
        selectedIndex = np.where(np.isin(vertexLabels, rndLabel))[0]
        remIndex = np.where(np.isin(vertexLabels, rndLabel, invert=True))[0]

        rndSelectedIndex = selectedIndex[
            np.random.randint(0, selectedIndex.shape[0], 1)
        ]

        if remIndex[0] == 0:
            return vertexLabels
        rndRemIndex = remIndex[np.random.randint(1, remIndex.shape[0], 1)]
        if self.tabuRule is not None:
            tabuList = self.tabuRule.getAvailableList()
            if not tabuList[rndSelectedIndex]:
                candidate = [x for x in range(1, vertexLabels.max() + 1)]
                while (~tabuList[selectedIndex]).all():
                    candidate.remove(rndLabel)
                    if len(candidate) == 0:
                        return vertexLabels
                    idx = np.random.randint(0, len(candidate), 1)
                    rndLabel = np.array(candidate[idx])
                    selectedIndex = np.where(np.isin(vertexLabels, rndLabel))[0]
                    remIndex = np.where(np.isin(vertexLabels, rndLabel, invert=True))[0]
            selectedIndex = selectedIndex[tabuList[selectedIndex]]
            remIndex = remIndex[tabuList[remIndex]]

            rndSelectedIndex = remIndex[np.random.randint(1, remIndex.shape[0], 1)]
            rndRemIndex = remIndex[np.random.randint(1, remIndex.shape[0], 1)]
            self.tabuRule.updateTabuList(rndSelectedIndex)
            self.tabuRule.updateTabuList(rndRemIndex)

        vertexLabels[rndSelectedIndex], vertexLabels[rndRemIndex] = (
            vertexLabels[rndRemIndex],
            vertexLabels[rndSelectedIndex],
        )

        return vertexLabels

    def closestSwap(self, vertexLabels):
        labelDisMatrix = self.calcLabelDisMatrix(vertexLabels)
        rndLabel = np.random.randint(0, vertexLabels.max() + 1, 1)
        *_, closestLabel = np.where(
            np.isin(labelDisMatrix[rndLabel, :], labelDisMatrix[rndLabel, :].min())
        )

        selectedIndex = np.where(np.isin(vertexLabels, rndLabel))[0]
        closestIndex = np.where(np.isin(vertexLabels, closestLabel))[0]

        rndSelectedIndex = selectedIndex[
            np.random.randint(0, selectedIndex.shape[0], 1)
        ]
        rndClosestIndex = closestIndex[np.random.randint(0, closestIndex.shape[0], 1)]

        if self.tabuRule is not None:
            tabuList = self.tabuRule.getAvailableList()
            if not tabuList[rndSelectedIndex]:
                candidate = [x for x in range(1, vertexLabels.max() + 1)]
                while (~tabuList[selectedIndex]).all() or (
                    ~tabuList[closestIndex].all()
                ):
                    candidate.remove(rndLabel)
                    if len(candidate) == 0:
                        return vertexLabels
                    idx = np.random.randint(0, len(candidate), 1)
                    rndLabel = np.array(candidate)[idx]
                    *_, closestLabel = np.where(
                        np.isin(
                            labelDisMatrix[rndLabel, :],
                            labelDisMatrix[rndLabel, :].min(),
                        )
                    )

                    selectedIndex = np.where(np.isin(vertexLabels, rndLabel))[0]
                    closestIndex = np.where(np.isin(vertexLabels, closestLabel))[0]
            selectedIndex = selectedIndex[tabuList[selectedIndex]]
            closestIndex = closestIndex[tabuList[closestIndex]]

            rndSelectedIndex = selectedIndex[
                np.random.randint(0, selectedIndex.shape[0], 1)
            ]
            rndClosestIndex = closestIndex[
                np.random.randint(0, closestIndex.shape[0], 1)
            ]
            self.tabuRule.updateTabuList(rndSelectedIndex)
            self.tabuRule.updateTabuList(rndClosestIndex)

        vertexLabels[rndSelectedIndex] = closestLabel
        vertexLabels[rndClosestIndex] = rndLabel

        return vertexLabels

    def remove2NewClusters(self, vertexLabels):
        rndIndex = np.random.randint(0, vertexLabels.max() + 1, 1)
        if self.tabuRule is not None:
            tabuList = self.tabuRule.getAvailableList()
            if not tabuList[rndIndex]:
                validIndex = [x for x in range(1, len(vertexLabels))]
                validIndex = np.array(validIndex)[tabuList[validIndex]].tolist()
                idx = np.random.randint(0, len(validIndex), 1)
                rndIndex = np.array(validIndex)[idx]
            self.tabuRule.updateTabuList(rndIndex)

        selectedLabel = vertexLabels[rndIndex]

        if np.where(np.isin(vertexLabels, selectedLabel))[0].size > 1:
            vertexLabels[rndIndex] = vertexLabels.max() + 1

        return vertexLabels
