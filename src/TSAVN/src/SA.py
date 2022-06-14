import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import timeRecording


class ConventionalParameter(object):
    def __init__(self, T, T_END, coolRate, maxIter):
        self.T = T
        self.T_END = T_END
        self.coolRate = coolRate
        self.maxIter = maxIter
        self.innerIt = 0

    def coolDown(self):
        self.T *= self.coolRate

    def isAcceptedSolution(self, dE):
        return not (dE > 0 and np.math.exp(-dE / self.T) < np.random.rand(1))

    def isReachCriteria(self):
        return self.T <= self.T_END

    def isReachInnerCriteria(self):
        flag = self.innerIt < self.maxIter
        if flag:
            self.innerIt += 1
        else:
            self.innerIt = 0
            self.coolDown()
        return flag


class AdaptiveParameter(object):
    def __init__(self, maxOutterIter, maxInnerIter, minTemp, weight, delta):
        self.maxOutterIter = maxOutterIter
        self.maxInnerIter = maxInnerIter
        self.minTemp = minTemp
        self.weight = weight
        self.delta = delta

        self.outterIter = 0
        self.innerIter = 0
        self.cnt = 0

    def isAcceptedSolution(self, dE):
        t = self.minTemp + self.weight * np.math.log(1 + self.cnt / self.delta)
        flag = dE > 0 and np.math.exp(-dE / t) < np.random.rand(1)
        self.updateCounter(dE)
        return not flag

    def updateCounter(self, dE):
        if dE > 0:
            self.cnt += 1
        elif dE < 0:
            self.cnt = 0

    def isReachCriteria(self):
        flag = self.outterIter < self.maxOutterIter
        self.outterIter += 1
        return not flag

    def isReachInnerCriteria(self):
        flag = self.innerIter < self.maxInnerIter
        if flag:
            self.innerIter += 1
        else:
            self.innerIter = 0
        return not flag


class SimulatedAnnealingAlgorithm(object):
    def __init__(self, objFunc, initSolution, feasibleFunc, optimizer, parameters):
        self.calcValue = objFunc
        self.currentSolution = initSolution()
        self.currentValue = objFunc(self.currentSolution)
        self.isFeasible = feasibleFunc
        self.optimizer = optimizer
        self.parameters = parameters

        self.globalBestSolution = None
        self.globalBestValue = np.inf

    @timeRecording
    def run(self):
        while not self.parameters.isReachCriteria():
            while not self.parameters.isReachInnerCriteria():
                self.step()

    def step(self):
        newSolution, newValue = self.optimizer(self.currentSolution, self.currentValue)

        dE = self.currentValue - newValue
        if self.parameters.isAcceptedSolution(dE):
            self.updateCurrentSolution(newSolution, newValue)

        if self.isGlobalBetterSolution(newSolution, newValue):
            self.updateGlobalBestSolution(newSolution, newValue)

    def updateCurrentSolution(self, newSolution, newValue):
        self.currentSolution = newSolution
        self.currentValue = newValue

    def isGlobalBetterSolution(self, newSolution, newValue):
        return self.isFeasible(newSolution) and self.globalBestValue > newValue

    def updateGlobalBestSolution(self, newSolution, newValue):
        self.globalBestSolution = newSolution
        self.globalBestValue = newValue


def main():
    pass


if __name__ == "__main__":
    main()
