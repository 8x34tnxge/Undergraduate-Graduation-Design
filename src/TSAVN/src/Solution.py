import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# set the 0th vertex is the original
class Solution(object):
    def __init__(self, uav_route, routine_cost):
        self.uav_route = uav_route
        self.routine_cost = routine_cost

    def __len__(self):
        return len(self.uav_route)
