import random
import pandas as pd
import argparse

from loguru import logger


def get_opt():
    parser = argparse.ArgumentParser(
        description="a tool to generate random data for resource assignment between uav and server (or edge computing)"
    )

    parser.add_argument(
        "-n",
        "--num",
        dest="NUM",
        default=10,
        type=int,
        help="the number of the mission to be assigned",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="OUTPUT_DIR",
        default="./data/data.csv",
        type=str,
        help="the output path for the generated data",
    )

    return parser.parse_args()


def generate_data(data_num: int) -> pd.DataFrame:
    id = [i for i in range(data_num)]
    start_time = [random.randint(0, 1000) for _ in range(data_num)]
    resource = [random.randint(1, 100) for _ in range(data_num)]
    runtime_in_uav = [random.randint(2, 300) for _ in range(data_num)]
    runtime_in_server = [
        random.randint(0, runtime_in_uav[i] - 1) for i in range(data_num)
    ]
    runtime_in_server = [
        runtime_in_uav[i] - random.randint(1, runtime_in_uav[i] - 1)
        for i in range(data_num)
    ]
    delay = [random.randint(1, 100) for _ in range(data_num)]

    return pd.DataFrame(
        data={
            "id": id,
            "start_time": start_time,
            "resource": resource,
            "runtime_in_uav": runtime_in_uav,
            "runtime_in_server": runtime_in_server,
            "delay": delay,
        }
    )


def save_data(df: pd.DataFrame, save_path: str = "./data/data.csv"):
    df.to_csv(save_path, index=False)


def main():
    opt = get_opt()
    df = generate_data(data_num=opt.NUM)
    save_data(df, opt.OUTPUT_DIR)
    logger.info("data was successfully generated!")


if __name__ == "__main__":
    main()
