import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from src.DataPreprocessor import DataPreprocessor
from src.DataSet import DataSet
from src.OptimizerSet import Optimizer
from src.SA import AdaptiveParameter, SimulatedAnnealingAlgorithm
from src.Solution import Solution
from src.TabuRules import TabuRules
from src.UAV import UAV_Data


def get_opt():
    parser = argparse.ArgumentParser(
        description="""
        this program is used to solve the vrp with max length (and time window)
        """
    )

    return parser.parse_args()

def dataPreprocess(n_clusters, position_data_path, save_dir, save_file_name):
    dataset = DataSet(UAV_Data(1, 1000), position_data_path)
    dataPreprocessor = DataPreprocessor(dataset.position_data)
    dataPreprocessor.fit(n_clusters)
    dataPreprocessor.save(save_dir, save_file_name)

def optim(position_data_path):
    dataset = DataSet(UAV_Data(1, 3000), position_data_path)
    tabuRules = TabuRules(2)
    optimizer = Optimizer(
        dataset,
        0.5,
        None
        # tabuRules
    )
    parameters = AdaptiveParameter(
        maxOutterIter=10,
        maxInnerIter=100,
        minTemp=1,
        weight=100,
        delta=10
    )
    algorithm = SimulatedAnnealingAlgorithm(
        dataset.objFunc,
        dataset.initSolution,
        dataset.feasibleFunc,
        optimizer.optimizer,
        parameters
    )
    algorithm.run()

    print(f'uav num:\t{len(algorithm.globalBestSolution)}')
    print(f'total fly dist:\t{algorithm.globalBestValue}')
    for uav_idx, uav_routine in enumerate(algorithm.globalBestSolution.uav_route):
        print(f'idx:\t{uav_idx}\t', end='')
        print('->'.join(map(str, uav_routine)))

    return dataset, algorithm.globalBestSolution, algorithm.globalBestValue

def main():
    opt = get_opt()
    n_clusters = 10
    data_path= './data/csv/c1_10_1.csv'
    dataset, solution, cost = optim(f'./data/csv/c1_10_1.csv')
    draw_points(solution ,dataset)
    plt.show()

def draw_points(solution,dataset):
    posXData = dataset.position_data['XCOORD.']
    posYData = dataset.position_data['YCOORD.']

    plt.scatter(dataset.position_data.loc[:, 'XCOORD.'], dataset.position_data.loc[:, 'YCOORD.'], s=8, linewidth=0.5, marker='x')
    
    for routine in solution.uav_route:
        posX = [posXData[0]]
        posY = [posYData[0]]
        for vertex in routine:
            posX.append(dataset.position_data.loc[vertex, 'XCOORD.'])
            posY.append(dataset.position_data.loc[vertex, 'YCOORD.'])
        posX.append(posXData[0])
        posY.append(posYData[0])
        plt.plot(posX, posY, linewidth=0.5)

    plt.xlabel('纬度')
    plt.ylabel('经度')


if __name__ == '__main__':
    main()
