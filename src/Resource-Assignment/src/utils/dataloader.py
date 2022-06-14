from collections import namedtuple

import numpy as np
import pandas as pd
from yacs.config import CfgNode

from utils.base import DataLoader


class CityDataLoader(DataLoader):
    def __init__(self, data: CfgNode) -> None:
        """the data loader containing the city data

        Args:
            data (CfgNode): the city data
        """
        coordPair = namedtuple("coordPair", ["x", "y"])
        processedData = dict()
        for elem in data:
            id = elem["id"]
            x = elem["cityCoord"][0]["x"]
            y = elem["cityCoord"][1]["y"]
            processedData[id] = coordPair(x, y)

        super().__init__(processedData)

        self.distMat = np.ones((len(data) + 1, len(data) + 1))
        for col, _ in self.data.items():
            for row, _ in self.data.items():
                if col > row:
                    continue

                elif col == row:
                    self.distMat[row][col] = 0
                    continue

                self.distMat[row][col] = np.linalg.norm(
                    np.array([self[col].x - self[row].x, self[col].y - self[row].y])
                )
                self.distMat[col][row] = np.linalg.norm(
                    np.array([self[col].x - self[row].x, self[col].y - self[row].y])
                )

class ResourceMissionDataLoader(DataLoader):
    def __init__(self, data_file: str) -> None:
        """the data loader containing the resource data

        Args:
            data (CfgNode): the resource data file
        """
        id = 0
        processedData = dict()
        df = pd.read_csv(data_file)
        for idx, row in df.iterrows():
            processedData[id] = row
            id += 1

        super().__init__(processedData)