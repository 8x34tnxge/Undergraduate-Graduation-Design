import os

from yacs.config import CfgNode

cwd = os.getcwd()
config = CfgNode(new_allowed=True)
config.merge_from_file(cwd+"/data/data_manager.yaml")
config.freeze()


def getConfig() -> CfgNode:
    return config
