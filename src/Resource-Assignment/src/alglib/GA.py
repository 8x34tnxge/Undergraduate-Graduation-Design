# from typing import Callable, List

# from tqdm import tqdm
# from utils.base import DataLoader
# from utils.tsp import crossover, distFunc, initSolution, mutate

# from alglib.base import Base
# from alglib.param import AC_Param


# class AntColonyAlgorithm(Base):
#     """
#     Ant Colony Algorithm
#     """

#     def __init__(self, params: AC_Param) -> None:
#         """initialize AC algorithm

#         Args:
#             params (AC_Param): the parameter object for Ant Colony Algorithm
#         """
#         self.bestValueWatcher = []
#         self.params: AC_Param = params
#         super().__init__()

#     def __setattr__(self, name: str, value: float) -> None:
#         """the method to set value watcher attribute

#         Args:
#             name (str): watcher's name
#             value (float): the notable value needed to be recorded
#         """
#         if name == "bestValue":
#             self.bestValueWatcher.append(value)
#         super().__setattr__(name, value)

#     def run(
#         self,
#         dataLoader: DataLoader,
#         initSchedule: Callable = initSolution,
#         calcValue: Callable = distFunc,
#         crossover: Callable = crossover,
#         mutate: Callable = mutate,
#     ) -> None:
#         """the main procedure for AC

#         Args:
#             dataLoader (DataLoader): where you can query data from
#             initSchedule (Callable, optional): the method to init schedule. Defaults to initSolution.
#             calcValue (Callable, optional): the method to calc value from given schedule. Defaults to distFunc.
#             crossover (Callable, optional): the method to let chromosomes crossover. Defaults to crossover.
#             mutate (Callable, optional): the method to let chromosomes mutate. Defaults to mutate.
#         """
#         # load params
#         epochNum = self.params.epochNum
#         popNum = self.params.popNum
#         maximize = self.params.maximize

#         # define the util functions
#         def updateLocalSchedule(schedule, value, maximize: bool = False) -> None:
#             """the method to update particle

#             Args:
#                 maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
#             """
#             if (self.localValue < value) ^ maximize:
#                 return

#             self.localSchedule = []
#             self.localSchedule.extend(schedule)
#             self.localValue = value

#         def updateGlobalSchedule(schedule, value, maximize: bool = False) -> None:
#             """the method to update global schedule

#             Args:
#                 maximize (bool, optional): whether to maximize or minimize the target value. Defaults to False.
#             """
#             if (self.bestValue < value) ^ maximize:
#                 return

#             self.bestSchedule = []
#             self.bestSchedule.extend(schedule)
#             self.bestValue = value

#         # main procedure