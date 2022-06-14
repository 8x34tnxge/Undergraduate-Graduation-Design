from typing import Callable, List

import numpy as np
from tqdm import tqdm
from utils.base import DataLoader
from utils.tsp import distFunc, initSolution, twoOpt

from alglib.base import Base
from alglib.param import AC_Param


class AntColonyAlgorithm(Base):
    """
    Ant Colony Algorithm
    """

    def __init__(self, params: AC_Param) -> None:
        """initialize AC algorithm

        Args:
            params (AC_Param): the parameter object for Ant Colony Algorithm
        """
        self.pheromones: np.array
        self.bestValueWatcher = []
        self.params: AC_Param = params
        super().__init__()

    def __setattr__(self, name: str, value: float) -> None:
        """the method to set value watcher attribute

        Args:
            name (str): watcher's name
            value (float): the notable value needed to be recorded
        """
        if name == "bestValue":
            self.bestValueWatcher.append(value)
        super().__setattr__(name, value)

    def run(
        self,
        dataLoader: DataLoader,
        initSchedule: Callable = initSolution,
        calcValue: Callable = distFunc,
        generateAnts: Callable = twoOpt,
    ) -> None:
        """the main procedure for AC

        Args:
            dataLoader (DataLoader): where you can query data from
            initSchedule (Callable, optional): the method to init schedule. Defaults to initSolution.
            calcValue (Callable, optional): the method to calc value from given schedule. Defaults to distFunc.
            generateAnts (Callable, optional): the method to generate Ants. Defaults to twoOpt.
        """
        # load params
        epochNum = self.params.epochNum
        popNum = self.params.popNum
        maximize = self.params.maximize

        # define the util functions

        def updateLocalSchedule(schedule, value, maximize: bool = False) -> None:
            """the method to update particle

            Args:
                maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
            """
            if (self.localValue < value) ^ maximize:
                return

            self.localSchedule, _ = initSchedule(dataLoader)
            self.localSchedule.machine_list = schedule.machine_list.copy()
            self.localSchedule.resource_list = schedule.resource_list.copy()
            self.localValue = value

        def updateGlobalSchedule(schedule, value, maximize: bool = False) -> None:
            """the method to update global schedule

            Args:
                maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
            """
            if (self.bestValue < value) ^ maximize:
                return

            self.bestSchedule, _ = initSchedule(dataLoader)
            self.bestSchedule.machine_list = schedule.machine_list.copy()
            self.bestSchedule.resource_list = schedule.resource_list.copy()
            self.bestValue = value

        # main procedure
        self.initPheromones()
        for epoch in tqdm(range(epochNum)):
            ants = generateAnts()
            self.updatePheromones(ants)
            for schedule, value in ants:
                updateLocalSchedule(schedule, value, maximize)
                updateGlobalSchedule(schedule, value, maximize)

    def initPheromones():
        pass

    def updatePheromones():
        pass
