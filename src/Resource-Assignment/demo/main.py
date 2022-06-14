import random
import time

import numpy as np
from loguru import logger
from src.alglib import SA, AdaptiveSA
from src.alglib.param import AdaptiveSA_Param, SA_Param
from src.config import getConfig
from src.utils import ResourceMissionDataLoader

#### penalty settings ####
OUT_MEMORY_PENALTY = 500
FINISH_PENALTY = 1
IDLE_PENALTY = 1

#### uav settings ####
UAV_NUM = 30
UAV_MAX_RESOURCE = 100

#### server settings ####
SERVER_NUM = 1
SERVER_MAX_RESOURCE = 2000


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


def calcValue(resource_list: np.ndarray):
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


def initSchedule(dataloader: ResourceMissionDataLoader, *args, **kwargs):
    mission_num = len(dataloader)
    schedule = Schedule(dataloader)
    machine_list = [
        random.randint(0, UAV_NUM + SERVER_NUM - 1) for _ in range(mission_num)
    ]
    for mission_id, machine in enumerate(machine_list):
        is_uav = 1 if machine in range(UAV_NUM) else 0
        resource = dataloader[mission_id]["resource"]
        start_time = dataloader[mission_id]["start_time"]
        runtime = (
            dataloader[mission_id]["runtime_in_uav"]
            if is_uav
            else dataloader[mission_id]["runtime_in_server"]
        )

        schedule.machine_list[mission_id] = machine
        schedule.resource_list[machine, start_time : start_time + runtime] += resource

    value = calcValue(schedule.resource_list)

    return schedule, value


def update_resource_list_exchange(
    dataloader, schedule, resource_list, prev_mission, post_mission
):
    prev_machine = schedule.machine_list[prev_mission]
    prev_resource = dataloader[prev_mission]["resource"]
    prev_start_time = dataloader[prev_mission]["start_time"]
    prev_runtime_in_uav = dataloader[prev_mission]["runtime_in_uav"]
    prev_runtime_in_server = dataloader[prev_mission]["runtime_in_server"]
    prev_runtime = (
        prev_runtime_in_uav
        if prev_machine in range(UAV_NUM)
        else prev_runtime_in_server
    )

    post_machine = schedule.machine_list[post_mission]
    post_resource = dataloader[post_mission]["resource"]
    post_start_time = dataloader[post_mission]["start_time"]
    post_runtime_in_uav = dataloader[post_mission]["runtime_in_uav"]
    post_runtime_in_server = dataloader[post_mission]["runtime_in_server"]
    post_runtime = (
        post_runtime_in_uav
        if post_machine in range(UAV_NUM)
        else post_runtime_in_server
    )

    # recalc prev machine
    resource_list[
        prev_machine, prev_start_time : prev_start_time + prev_runtime
    ] -= prev_resource
    resource_list[
        prev_machine, post_start_time : post_start_time + post_runtime
    ] += post_resource

    # recalc post machine
    resource_list[
        post_machine, post_start_time : post_start_time + post_runtime
    ] -= post_resource
    resource_list[
        post_machine, prev_start_time : prev_start_time + prev_runtime
    ] += prev_resource

    return resource_list


def exchange(schedule: Schedule, dataloader, value, *args, **kwargs):
    best_delta = np.inf
    best_mission_list = []
    mission_num = schedule.machine_list.shape[0]
    for i in range(mission_num - 1):
        for k in range(i + 1, mission_num):
            resource_list = schedule.resource_list.copy()
            resource_list = update_resource_list_exchange(
                dataloader, schedule, resource_list, i, k
            )

            if schedule.machine_list[i] == schedule.machine_list[k]:
                continue

            new_value = calcValue(resource_list)
            delta = new_value - value
            if delta < best_delta:
                best_mission_list = (i, k)
                best_delta = delta

    if best_delta == np.inf:
        return schedule, value
    schedule.resource_list = update_resource_list_exchange(
        dataloader, schedule, schedule.resource_list, *best_mission_list
    )
    (
        schedule.machine_list[best_mission_list[0]],
        schedule.machine_list[best_mission_list[1]],
    ) = (
        schedule.machine_list[best_mission_list[1]],
        schedule.machine_list[best_mission_list[0]],
    )

    return schedule, value + best_delta


def update_resource_list_uav2uav(
    dataloader, schedule, resource_list, prev_mission, post_machine
):
    prev_machine = schedule.machine_list[prev_mission]
    prev_resource = dataloader[prev_mission]["resource"]
    prev_start_time = dataloader[prev_mission]["start_time"]
    prev_runtime_in_uav = dataloader[prev_mission]["runtime_in_uav"]

    # recalc prev machine
    resource_list[
        prev_machine, prev_start_time : prev_start_time + prev_runtime_in_uav
    ] -= prev_resource

    # recalc post machine
    resource_list[
        post_machine, prev_start_time : prev_start_time + prev_runtime_in_uav
    ] += prev_resource

    return resource_list


def uav2uav(schedule, dataloader, value, *args, **kwargs):
    best_delta = np.inf
    best_device = np.inf
    best_mission = np.inf
    mission_num = schedule.machine_list.shape[0]
    for i in range(mission_num - 1):
        prev_machine = schedule.machine_list[i]
        if prev_machine not in range(UAV_NUM):
            continue
        for post_machine in range(UAV_NUM):
            resource_list = schedule.resource_list.copy()

            if prev_machine == post_machine:
                continue

            resource_list = update_resource_list_uav2uav(
                dataloader, schedule, resource_list, i, post_machine
            )

            new_value = calcValue(resource_list)
            delta = new_value - value
            if delta < best_delta:
                best_mission = i
                best_device = post_machine
                best_delta = delta

    if best_delta == np.inf:
        return schedule, value
    schedule.resource_list = update_resource_list_uav2uav(
        dataloader, schedule, schedule.resource_list, best_mission, best_device
    )
    schedule.machine_list[best_mission] = best_device

    return schedule, value + best_delta


def update_resource_list_uav2server(
    dataloader, schedule, resource_list, prev_mission, post_machine
):
    prev_machine = schedule.machine_list[prev_mission]
    prev_resource = dataloader[prev_mission]["resource"]
    prev_start_time = dataloader[prev_mission]["start_time"]
    prev_runtime_in_uav = dataloader[prev_mission]["runtime_in_uav"]
    prev_runtime_in_server = dataloader[prev_mission]["runtime_in_server"]

    # recalc prev machine
    resource_list[
        prev_machine, prev_start_time : prev_start_time + prev_runtime_in_uav
    ] -= prev_resource

    # recalc post machine
    resource_list[
        post_machine, prev_start_time : prev_start_time + prev_runtime_in_server
    ] += prev_resource

    return resource_list


def uav2server(schedule, dataloader, value, prob):
    best_delta = np.inf
    best_mission = np.inf
    mission_num = schedule.machine_list.shape[0]
    server_idx = UAV_NUM + SERVER_NUM - 1
    for i in range(mission_num):
        resource_list = schedule.resource_list.copy()

        prev_machine = schedule.machine_list[i]
        if prev_machine not in range(UAV_NUM):
            continue

        resource_list = update_resource_list_uav2server(
            dataloader, schedule, resource_list, i, server_idx
        )

        new_value = calcValue(resource_list)
        delta = new_value - value
        if delta < best_delta:
            best_mission = i
            best_delta = delta

    if best_delta == np.inf:
        return schedule, value
    schedule.resource_list = update_resource_list_uav2server(
        dataloader, schedule, schedule.resource_list, best_mission, server_idx
    )
    schedule.machine_list[best_mission] = server_idx

    return schedule, value + best_delta


def update_resource_list_server2uav(
    dataloader, schedule, resource_list, prev_mission, post_machine
):
    prev_machine = schedule.machine_list[prev_mission]
    prev_resource = dataloader[prev_mission]["resource"]
    prev_start_time = dataloader[prev_mission]["start_time"]
    prev_runtime_in_uav = dataloader[prev_mission]["runtime_in_uav"]
    prev_runtime_in_server = dataloader[prev_mission]["runtime_in_server"]

    # recalc prev machine
    resource_list[
        prev_machine, prev_start_time : prev_start_time + prev_runtime_in_server
    ] -= prev_resource

    # recalc post machine
    resource_list[
        post_machine, prev_start_time : prev_start_time + prev_runtime_in_uav
    ] += prev_resource

    return resource_list


def server2uav(schedule, dataloader, value, prob):
    best_delta = np.inf
    best_mission = np.inf
    best_device = np.inf
    mission_num = schedule.machine_list.shape[0]
    for i in range(mission_num - 1):
        prev_machine = schedule.machine_list[i]
        if prev_machine in range(UAV_NUM):
            continue
        for post_machine in range(UAV_NUM):
            resource_list = schedule.resource_list.copy()
            resource_list = update_resource_list_server2uav(
                dataloader, schedule, resource_list, i, post_machine
            )

            new_value = calcValue(resource_list)
            delta = new_value - value
            if delta < best_delta:
                best_mission = i
                best_device = post_machine
                best_delta = delta

    if best_delta == np.inf:
        return schedule, value
    prev_machine = schedule.machine_list[best_mission]
    post_machine = best_device

    schedule.resource_list = update_resource_list_server2uav(
        dataloader, schedule, schedule.resource_list, best_mission, best_device
    )
    schedule.machine_list[best_mission] = best_device

    return schedule, value + best_delta


methods = (
    exchange,
    uav2uav,
    uav2server,
    server2uav,
)
scores = [1 for _ in range(len(methods))]


def fetchNewSchedule(self, localSchedule, localValue, dataloader, *args, **kwargs):
    weight = [score / sum(scores) for score in scores]
    method_idx = np.random.choice(
        [idx for idx in range(len(methods))],
        size=1,
        replace=False,
        p=weight,
    )[0]
    new_schedule, new_value = methods[method_idx](
        localSchedule, dataloader, localValue, 0.2
    )
    if new_value < localValue:
        scores[method_idx] += sum(scores) / len(scores)
    else:
        scores[method_idx] *= 0.8
        scores[method_idx] = max(1e-6, scores[method_idx])

    return new_schedule, new_value


def main():
    cfg = getConfig()

    SA_values = []
    SA_runtimes = []
    AdaptiveSA_values = []
    AdaptiveSA_runtimes = []

    for data_file in cfg["DataFile"]:
        logger.info(data_file)
        dataloader = ResourceMissionDataLoader(data_file)
        params = SA_Param(
            initialTemperate=1e4,
            terminatedTemperate=1e0,
            coolRate=0.9,
            epochNum=15,
            method=None,
            maximize=False,
            doIntersectAnalysis=None,
            initStatusJudgement=0,
        )
        alg = SA(params)
        start = time.time()
        alg.run(dataloader, initSchedule, calcValue, fetchNewSchedule)
        duration = time.time() - start
        SA_values.append(alg.bestValue)
        logger.info(f"best value: {alg.bestValue}")
        SA_runtimes.append(duration)
        logger.info(f"runtime: {duration}")

        params = AdaptiveSA_Param(
            epochNum=1000,
            minTemperate=1,
            penalWeight=1,
            delta=1e-2,
            method=None,
            maximize=False,
            doIntersectAnalysis=None,
            initStatusJudgement=0,
        )
        alg = AdaptiveSA(params)
        start = time.time()
        alg.run(dataloader, initSchedule, calcValue, fetchNewSchedule)
        duration = time.time() - start
        AdaptiveSA_values.append(alg.bestValue)
        logger.info(f"best value: {alg.bestValue}")
        AdaptiveSA_runtimes.append(duration)
        logger.info(f"runtime: {duration}")

    SA_values = np.array(SA_values)
    SA_runtimes = np.array(SA_runtimes)
    AdaptiveSA_values = np.array(AdaptiveSA_values)
    AdaptiveSA_runtimes = np.array(AdaptiveSA_runtimes)

    logger.info(SA_values)
    logger.info(SA_runtimes)
    logger.info(
        f"Algorithm: SA value mean: {SA_values.mean()} value std: {SA_values.std()} runtime mean: {SA_runtimes.mean()} runtime std: {SA_runtimes.std()}"
    )
    logger.info(AdaptiveSA_values)
    logger.info(AdaptiveSA_runtimes)
    logger.info(
        f"Algorithm: AdaptiveSA value mean: {AdaptiveSA_values.mean()} value std: {AdaptiveSA_values.std()} runtime mean: {AdaptiveSA_runtimes.mean()} runtime std: {AdaptiveSA_runtimes.std()}"
    )


if __name__ == "__main__":
    main()
