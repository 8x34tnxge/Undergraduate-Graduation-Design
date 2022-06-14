from dataclasses import dataclass

import numpy as np


@dataclass
class AntColonyAlgorithmParamSetting:
    popNum: np.int16
    alpha: float
    beta: float
    sigma: float
    increasement: float
    epochNum: np.int16
    maximize: bool
