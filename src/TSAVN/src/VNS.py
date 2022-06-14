import copy
from dataclasses import replace
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from src.Solution import Solution

logger.remove(0)
logger.add("test.log")


class VariableNeighborhoodSearch(object):
    def __init__(self, label_idx, disMatrix, is_feasible, maxIteration=1):
        self.label_idx = label_idx
        self.disMatrix = disMatrix
        self.is_feasible = is_feasible
        solution = self.init_solution()
        self.solution = Solution(solution, self.calcCost(solution))
        self.maxIteration = maxIteration
        self.precision = 1e-6

    def init_solution(self):
        ret = []
        routine = []
        for idx in self.label_idx:
            routine.append(idx)
            if not self.is_feasible(routine):
                routine.pop()
                ret.append(routine)
                routine = [idx]

        if self.is_feasible(routine):
            ret.append(routine)

        return ret

    def calcCost(self, solution):
        cost = 0
        for routine in solution:
            if len(routine) == 0:
                continue
            cost += self.disMatrix[0, routine[0]] + self.disMatrix[0, routine[-1]]
            for i in range(1, len(routine)):
                cost += self.disMatrix[routine[i - 1], routine[i]]
        return cost

    def calcDiff1(self, i, j, solution):
        """
        修正路径长度（时间复杂度O(1)）
        :param i: 前改变点位置
        :param j: 后改变点位置
        :param solution: 城市序列
        :return: 修正后路径差
        """
        delta = 0
        if i == 0 and j == len(solution) - 1:
            return delta

        if i == 0:
            delta -= self.disMatrix[0, solution[i]]
            delta += self.disMatrix[0, solution[j]]
        else:
            delta -= self.disMatrix[solution[i - 1], solution[i]]
            delta += self.disMatrix[solution[i - 1], solution[j]]

        if j == len(solution) - 1:
            delta -= self.disMatrix[0, solution[j]]
            delta += self.disMatrix[0, solution[i]]
        else:
            delta -= self.disMatrix[solution[j + 1], solution[j]]
            delta += self.disMatrix[solution[j + 1], solution[i]]

        return delta

    def twoOpt(self, i, j, solution):
        ret = []
        ret.extend(solution[:i])

        for idx in range(j, i - 1, -1):
            ret.append(solution[idx])

        ret.extend(solution[j + 1 :])

        return ret

    def neighborhood_one(self):
        """
        邻域结构 一
        :param current_solution: 当前解
        :return: 新解
        """
        flag = False
        for count in range(self.maxIteration):
            for idx in range(len(self.solution.uav_route)):
                # if len(routine) <= 2:
                #     continue
                # i, j = np.random.choice(range(len(routine)), 2, replace=False)
                # i, j = min(i, j), max(i, j)
                for i in range(len(self.solution.uav_route[idx])):
                    for j in range(i + 1, len(self.solution.uav_route[idx])):
                        routine = self.solution.uav_route[idx]
                        delta = self.calcDiff1(i, j, routine)
                        if delta < -self.precision:
                            logger.debug(i)
                            logger.debug(j)
                            logger.debug(delta)
                            logger.debug(routine)
                            logger.debug(self.calcCost([routine]))
                            self.solution.uav_route[idx] = self.twoOpt(i, j, routine)
                            logger.debug(self.solution.uav_route[idx])
                            logger.debug(self.calcCost([self.solution.uav_route[idx]]))
                            self.solution.routine_cost += delta
                            flag = True
        return flag

    def calcDiff2(self, i, j, solution):
        """
        修正路径长度（时间复杂度O(1)）
        :param i: 前改变点位置
        :param j: 后改变点位置
        :param solution: 城市序列
        :return: 修正后路径差
        """
        delta = 0
        if i == j or i == j - 1 or i > j:
            return delta
        elif j == len(solution) - 1:
            delta -= self.disMatrix[solution[i], solution[i + 1]]
            delta -= self.disMatrix[solution[j], solution[j - 1]]
            delta -= self.disMatrix[0, solution[j]]
            delta += self.disMatrix[0, solution[j - 1]]
            delta += self.disMatrix[solution[i], solution[j]]
            delta += self.disMatrix[solution[j], solution[i + 1]]
        else:
            delta -= self.disMatrix[solution[i], solution[i + 1]]
            delta -= self.disMatrix[solution[j], solution[j - 1]]
            delta -= self.disMatrix[solution[j + 1], solution[j]]
            delta += self.disMatrix[solution[j + 1], solution[j - 1]]
            delta += self.disMatrix[solution[i], solution[j]]
            delta += self.disMatrix[solution[j], solution[i + 1]]
        return delta

    def twoHOpt(self, i, j, solution):
        """
        将j位置的元素插入到i位置的元素后面
        :param i: 前改变点
        :param j: 后改变点
        :param current_solution: 当前解
        """
        ret = []
        ret.extend(solution)

        tmp = solution[j]
        ret.remove(tmp)
        if j > i:
            ret.insert(i + 1, tmp)
        else:
            ret.insert(i, tmp)

        return ret

    def neighborhood_two(self):
        """
        邻域结构 一
        :param current_solution: 当前解
        :return: 新解
        """
        flag = False
        for count in range(self.maxIteration):
            # for idx, routine in enumerate(self.solution.uav_route):
            for idx in range(len(self.solution.uav_route)):
                # if len(routine) <= 2:
                #     continue
                # i, j = np.random.choice(range(len(routine)), 2, replace=False)
                # i, j = min(i, j), max(i, j)
                for i in range(len(self.solution.uav_route[idx])):
                    for j in range(len(self.solution.uav_route[idx])):
                        routine = self.solution.uav_route[idx]
                        if i == j:
                            continue
                        delta = self.calcDiff2(i, j, routine)
                        if delta < -self.precision:
                            self.solution.uav_route[idx] = self.twoHOpt(i, j, routine)
                            self.solution.routine_cost += delta
                            flag = True
        return flag

    def transfer(self, i, j, solution_before: List, solution_after: List):
        """_summary_

        Parameters
        ----------
        i : _type_
            the pre-index of vertex
        j : _type_
            the post-index of vertex
        solution_before : _type_
            the solution before
        solution_after : _type_
            the solution after
        """
        solution1 = []
        solution2 = []
        solution1.extend(solution_before)
        solution2.extend(solution_after)

        vertex = solution1.pop(i)
        solution2.insert(j, vertex)
        return solution1, solution2

    def neighborhood_three(self):
        flag = False
        for count in range(self.maxIteration):
            for idx1 in range(len(self.solution.uav_route)):
                for idx2 in range(len(self.solution.uav_route)):
                    routine1 = self.solution.uav_route[idx1]
                    routine2 = self.solution.uav_route[idx2]
                    if idx1 == idx2:
                        continue
                    if len(routine1) == 0:
                        continue
                    if len(routine2) == 0:
                        continue
                    # i = np.random.randint(0, len(routine1))
                    # j = np.random.randint(0, len(routine2))
                    for i in range(len(routine1)):
                        for j in range(len(routine2)):
                            delta = self.calcDiff3(i, j, routine1, routine2)
                            if delta < -self.precision:
                                tmp1, tmp2 = self.transfer(i, j, routine1, routine2)
                                if not self.is_feasible(
                                    self.calcCost([tmp1])
                                ) or not self.is_feasible(self.calcCost([tmp2])):
                                    continue
                                (
                                    self.solution.uav_route[idx1],
                                    self.solution.uav_route[idx2],
                                ) = (tmp1, tmp2)
                                self.solution.routine_cost += delta
                                flag = True
                                break
                        if delta < -self.precision:
                            break
        return flag

    def calcDiff3(self, i, j, solution1, solution2):
        delta = 0
        vertex = solution1[i]
        if i == 0 and i == len(solution1) - 1:
            delta -= 2 * self.disMatrix[0, vertex]
        elif i == 0:
            delta -= self.disMatrix[0, vertex]
            delta -= self.disMatrix[vertex, solution1[i + 1]]
            delta += self.disMatrix[0, solution1[i + 1]]
        elif i == len(solution1) - 1:
            delta -= self.disMatrix[solution1[i - 1], vertex]
            delta -= self.disMatrix[0, vertex]
            delta += self.disMatrix[solution1[i - 1], 0]
        else:
            delta -= self.disMatrix[solution1[i - 1], vertex]
            delta -= self.disMatrix[vertex, solution1[i + 1]]
            delta += self.disMatrix[solution1[i - 1], solution1[i + 1]]

        if j == 0 and j == len(solution2):
            delta += 2 * self.disMatrix[0, vertex]
        elif j == 0:
            delta -= self.disMatrix[0, solution2[j]]
            delta += self.disMatrix[0, vertex]
            delta += self.disMatrix[vertex, solution2[j]]
        elif j == len(solution2):
            delta -= self.disMatrix[solution2[j - 1], 0]
            delta += self.disMatrix[solution2[j - 1], vertex]
            delta += self.disMatrix[0, vertex]
        else:
            delta -= self.disMatrix[solution2[j - 1], solution2[j]]
            delta += self.disMatrix[solution2[j - 1], vertex]
            delta += self.disMatrix[vertex, solution2[j]]

        return delta

    def exchange(self, i, j, solution1, solution2):
        solution_before = []
        solution_after = []
        solution_before.extend(solution1)
        solution_after.extend(solution2)

        solution_before[i], solution_after[j] = solution_after[j], solution_before[i]
        return solution_before, solution_after

    def neighborhood_four(self):
        flag = False
        for count in range(self.maxIteration):
            for idx1 in range(len(self.solution.uav_route)):
                for idx2 in range(len(self.solution.uav_route)):
                    routine1 = self.solution.uav_route[idx1]
                    routine2 = self.solution.uav_route[idx2]
                    if idx1 == idx2:
                        continue
                    if len(routine1) == 0:
                        continue
                    if len(routine2) == 0:
                        continue
                    # i = np.random.randint(0, len(routine1))
                    # j = np.random.randint(0, len(routine2))
                    for i in range(len(routine1)):
                        for j in range(len(routine2)):
                            delta = self.calcDiff4(i, j, routine1, routine2)
                            if delta < -self.precision:
                                tmp1, tmp2 = self.exchange(i, j, routine1, routine2)
                                if not self.is_feasible(
                                    self.calcCost([tmp1])
                                ) or not self.is_feasible(self.calcCost([tmp2])):
                                    continue
                                (
                                    self.solution.uav_route[idx1],
                                    self.solution.uav_route[idx2],
                                ) = (tmp1, tmp2)
                                self.solution.routine_cost += delta
                                flag = True
        return flag

    def calcDiff4(self, i, j, solution1, solution2):
        delta = 0
        vertex1 = solution1[i]
        if i == 0 and i == len(solution1) - 1:
            delta -= 2 * self.disMatrix[0, vertex1]
        elif i == 0:
            delta -= self.disMatrix[0, vertex1]
            delta -= self.disMatrix[vertex1, solution1[i + 1]]
        elif i == len(solution1) - 1:
            delta -= self.disMatrix[solution1[i - 1], vertex1]
            delta -= self.disMatrix[0, vertex1]
        else:
            delta -= self.disMatrix[solution1[i - 1], vertex1]
            delta -= self.disMatrix[vertex1, solution1[i + 1]]

        if j == 0 and j == len(solution2) - 1:
            delta += 2 * self.disMatrix[0, vertex1]
        elif j == 0:
            delta += self.disMatrix[0, vertex1]
            delta += self.disMatrix[vertex1, solution2[j + 1]]
        elif j == len(solution2) - 1:
            delta += self.disMatrix[solution2[j - 1], vertex1]
            delta += self.disMatrix[0, vertex1]
        else:
            delta += self.disMatrix[solution2[j - 1], vertex1]
            delta += self.disMatrix[vertex1, solution2[j + 1]]

        vertex2 = solution2[j]
        if i == 0 and i == len(solution1) - 1:
            delta += 2 * self.disMatrix[0, vertex2]
        elif i == 0:
            delta += self.disMatrix[0, vertex2]
            delta += self.disMatrix[vertex2, solution1[i + 1]]
        elif i == len(solution1) - 1:
            delta += self.disMatrix[solution1[i - 1], vertex2]
            delta += self.disMatrix[0, vertex2]
        else:
            delta += self.disMatrix[solution1[i - 1], vertex2]
            delta += self.disMatrix[vertex2, solution1[i + 1]]

        if j == 0 and j == len(solution2) - 1:
            delta -= 2 * self.disMatrix[0, vertex2]
        elif j == 0:
            delta -= self.disMatrix[0, vertex2]
            delta -= self.disMatrix[vertex2, solution2[j + 1]]
        elif j == len(solution2) - 1:
            delta -= self.disMatrix[solution2[j - 1], vertex2]
            delta -= self.disMatrix[0, vertex2]
        else:
            delta -= self.disMatrix[solution2[j - 1], vertex2]
            delta -= self.disMatrix[vertex2, solution2[j + 1]]

        return delta

    def add_new(self, i, solution):
        origin_solution = []
        new_solution = []
        origin_solution.extend(solution)
        vertex = origin_solution.pop(i)
        new_solution.append(vertex)

        return origin_solution, new_solution

    def neighborhood_five(self):
        flag = False
        for count in range(self.maxIteration):
            # for idx, routine in enumerate(self.solution.uav_route):
            for idx in range(len(self.solution.uav_route)):
                routine = self.solution.uav_route[idx]
                # i = np.random.randint(0, len(routine))
                for i in range(len(routine)):
                    delta = self.calcDiff5(i, routine)
                    if delta < -self.precision:
                        tmp1, tmp2 = self.add_new(i, routine)
                        if not self.is_feasible(
                            self.calcCost([tmp1])
                        ) or not self.is_feasible(self.calcCost([tmp2])):
                            continue
                        flag = True
                        self.solution.uav_route[idx], new_routine = tmp1, tmp2
                        self.solution.uav_route.append(new_routine)
                        self.solution.routine_cost += delta
                        break
        return flag

    def calcDiff5(self, i, solution):
        delta = 0
        vertex = solution[i]
        if i == 0 and i == len(solution) - 1:
            delta -= 2 * self.disMatrix[0, vertex]
        elif i == 0:
            delta -= self.disMatrix[0, vertex]
            delta -= self.disMatrix[vertex, solution[i + 1]]
            delta += self.disMatrix[0, solution[i + 1]]
        elif i == len(solution) - 1:
            delta -= self.disMatrix[solution[i - 1], vertex]
            delta -= self.disMatrix[0, vertex]
            delta += self.disMatrix[solution[i - 1], 0]
        else:
            delta -= self.disMatrix[solution[i - 1], vertex]
            delta -= self.disMatrix[vertex, solution[i + 1]]
            delta += self.disMatrix[solution[i - 1], solution[i + 1]]

        delta += 2 * self.disMatrix[0, vertex]
        return delta

    def fit(self):
        cnt = 1
        while cnt < 6:
            if cnt == 1:
                status = self.neighborhood_one()
                if status is True:
                    cnt = 0
            if cnt == 2:
                status = self.neighborhood_two()
                if status is True:
                    cnt = 0
            if cnt == 3:
                status = self.neighborhood_three()
                if status is True:
                    cnt = 0
            if cnt == 4:
                status = self.neighborhood_four()
                if status is True:
                    cnt = 0
            if cnt == 5:
                status = self.neighborhood_five()
                if status is True:
                    cnt = 0
            new_routine = []
            for routine in self.solution.uav_route:
                if len(routine) == 0:
                    continue
                new_routine.append(routine)
            self.solution.uav_route = new_routine

            cnt += 1
        return self.solution
