import enum
import time
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from src.config import getConfig
from src.utils import ResourceMissionDataLoader

#### penalty settings ####
OUT_MEMORY_PENALTY = 10000
FINISH_PENALTY = 1
IDLE_PENALTY = 1

#### uav settings ####
UAV_NUM = 30
UAV_MAX_RESOURCE = 100

#### server settings ####
SERVER_NUM = 1
SERVER_MAX_RESOURCE = 2000


class ScheduleStatus(enum.Enum):
    Success = enum.auto()
    Failure = enum.auto()


class Schedule:
    def __init__(self, dataloader: ResourceMissionDataLoader):
        self.machine_list = self.init_machine_list(dataloader)
        self.resource_list = self.init_resource_list(dataloader)

    def init_machine_list(self, dataloader: ResourceMissionDataLoader):
        return np.zeros([len(dataloader)], dtype=np.int16)

    def init_resource_list(self, dataloader: ResourceMissionDataLoader):
        max_finish_time = 0
        for mission_id, mission_info in dataloader:
            start_time = mission_info["start_time"]
            max_runtime = mission_info["runtime_in_uav"]
            max_finish_time = max(max_finish_time, start_time + max_runtime)

        return np.zeros((UAV_NUM + SERVER_NUM, max_finish_time))


def move2uav(
    schedule: Schedule, mission_info: pd.Series, uav_idx
) -> Tuple[ScheduleStatus, float]:
    start_time = mission_info["start_time"]
    runtime_in_uav = mission_info["runtime_in_uav"]

    resource_list = schedule.resource_list[
        uav_idx, start_time : start_time + runtime_in_uav
    ]

    # if resource_list.max() + resource > UAV_MAX_RESOURCE:
    #     return ScheduleStatus.Failure, np.inf

    idle_time = np.isin(resource_list, [0]).sum()
    return (
        ScheduleStatus.Success,
        runtime_in_uav * FINISH_PENALTY - idle_time * IDLE_PENALTY,
    )


def move2server(
    schedule: Schedule, mission_info: pd.Series
) -> Tuple[ScheduleStatus, float]:
    server_idx = UAV_NUM + SERVER_NUM - 1

    start_time = mission_info["start_time"]
    runtime_in_server = mission_info["runtime_in_server"]

    resource_list = schedule.resource_list[
        server_idx, start_time : start_time + runtime_in_server
    ]

    # if resource_list.max() + resource > SERVER_MAX_RESOURCE:
    #     return ScheduleStatus.Failure, np.inf

    idle_time = np.isin(resource_list, [0]).sum()
    return (
        ScheduleStatus.Success,
        runtime_in_server * FINISH_PENALTY - idle_time * IDLE_PENALTY,
    )


def assign_device(
    schedule: Schedule,
    dataloader: ResourceMissionDataLoader,
    mission_id: int,
    device_idx: int,
):
    mission_info = dataloader[mission_id]
    resource = mission_info["resource"]
    start_time = mission_info["start_time"]
    runtime = (
        mission_info["runtime_in_uav"]
        if device_idx < UAV_NUM
        else mission_info["runtime_in_server"]
    )

    schedule.machine_list[mission_id] = device_idx
    schedule.resource_list[device_idx, start_time : start_time + runtime] += resource

    return schedule


def calcValue(schedule: Schedule):
    resource_list = schedule.resource_list

    # calc finish time
    _, mission_finish_time = np.nonzero(resource_list)
    finish_time = mission_finish_time.max()

    # calc idle time
    idle_time = np.isin(resource_list, [0]).sum()

    # calc the times of resource out
    uav_resource_outnumber_num = (resource_list[0:UAV_NUM, :] > UAV_MAX_RESOURCE).sum()
    server_resource_outnumber_num = (
        resource_list[UAV_NUM, :] > SERVER_MAX_RESOURCE
    ).sum()

    return (
        finish_time * FINISH_PENALTY
        + idle_time * IDLE_PENALTY
        + (uav_resource_outnumber_num + server_resource_outnumber_num)
        * OUT_MEMORY_PENALTY
    )


def main():
    cfg = getConfig()
    values = []
    durations = []

    for data_file in cfg["DataFile"]:
        dataloader = ResourceMissionDataLoader(data_file)
        start = time.time()
        schedule = Schedule(dataloader)
        for mission_id, mission_info in dataloader:
            # init params
            best_device_idx, best_delta = -1, np.inf

            # check uav
            for uav_idx in range(UAV_NUM):
                status, delta = move2uav(schedule, mission_info, uav_idx)
                if status is ScheduleStatus.Success and delta < best_delta:
                    best_device_idx = uav_idx
                    best_delta = delta

            # check server
            status, delta = move2server(schedule, mission_info)
            if status is ScheduleStatus.Success and delta < best_delta:
                best_device_idx = UAV_NUM + SERVER_NUM - 1
                best_delta = delta

            # assign device
            schedule = assign_device(schedule, dataloader, mission_id, best_device_idx)
        duration = time.time() - start
        durations.append(duration)

        resource_list = schedule.resource_list
        uav_resource_outnumber_num = (
            resource_list[0:UAV_NUM, :] > UAV_MAX_RESOURCE
        ).sum()
        server_resource_outnumber_num = (
            resource_list[UAV_NUM, :] > SERVER_MAX_RESOURCE
        ).sum()
        logger.info(uav_resource_outnumber_num)
        logger.info(server_resource_outnumber_num)

        value = calcValue(schedule)
        logger.info(value)
        values.append(value)
        logger.info(f"runtime: {duration}")
    values = np.array(values)
    logger.info(f"values: {values}, mean: {values.mean()}, std: {values.std()}")
    durations = np.array(durations)
    logger.info(durations)
    logger.info(
        f"durations: {durations}, mean: {durations.mean()}, std: {durations.std()}"
    )


if __name__ == "__main__":
    main()
