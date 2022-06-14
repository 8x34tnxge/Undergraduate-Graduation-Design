import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TabuRules(object):
    def __init__(self, tabuStepLength):
        self.tabuStepLength = tabuStepLength

    def initialize(self, vertexNum):
        self.iter = 0
        self.tabuList = np.zeros(vertexNum)

    def updateIter(self):
        self.iter += 1

    def getAvailableList(self):
        self.updateIter()
        return self.tabuList <= self.iter

    def updateTabuList(self, index):
        self.tabuList[index] += self.tabuStepLength


def main():
    pass


if __name__ == "__main__":
    main()
