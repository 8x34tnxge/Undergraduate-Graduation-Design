import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class DataPreprocessor(object):
    def __init__(self, positionData):
        self.positionData = positionData
        self.label = np.zeros(positionData.shape[0], dtype=np.int)

    def fit(self, n_clusters):
        model = KMeans(n_clusters=n_clusters)
        data = self.positionData.loc[1:, ["XCOORD.", "YCOORD."]].to_numpy()
        model.fit(data)
        self.label[1:] = model.predict(data)

    def save(self, path, file_name):
        for idx in range(self.label.max() + 1):
            self.label[0] = idx
            self.positionData.loc[np.where(np.isin(self.label, idx))].to_csv(
                os.path.join(path, f"{file_name}_{idx}.csv"), index=0
            )
