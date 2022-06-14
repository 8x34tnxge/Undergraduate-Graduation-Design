import argparse
import os

import pandas as pd


def get_opt():
    parser = argparse.ArgumentParser(
        """
        the program to process the vrp datasets into a standard csv files
        """
    )

    parser.add_argument(
        "-d",
        "--dir",
        dest="dir",
        required=True,
        default="./data",
        help="the dir of the datasets",
    )

    parser.add_argument("-o", "--output_dir", dest="output_dir", default="./data/csv")

    return parser.parse_args()


def process(file_name: str, output_dir):
    with open(file_name, "r") as f:
        dataset_name = f.readline().rsplit("\n")[0]
        txt = f.readline()
        while txt != "CUSTOMER\n":
            txt = f.readline()
        col_names = f.readline()
        col_names = list(
            map(
                lambda x: x.strip(" "),
                filter(lambda x: x != "", col_names.rstrip("\n").split("  ")),
            )
        )

        # remove the empty line
        txt = f.readline()

        customer_info = {col_name: [] for col_name in col_names}
        txt = f.readline()
        while txt != "":
            customer = list(
                map(
                    lambda x: x.strip(" "),
                    filter(lambda x: x != "", txt.rstrip("\n").split(" ")),
                )
            )
            txt = f.readline()
            for col_idx, col_name in enumerate(col_names):
                customer_info[col_name].append(customer[col_idx])

        db = pd.DataFrame(customer_info)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        db.to_csv(os.path.join(output_dir, ".".join([dataset_name, "csv"])))


def main():
    opt = get_opt()
    dataset_names = os.listdir(opt.dir)
    for dataset_name in dataset_names:
        process(os.path.join(opt.dir, dataset_name), opt.output_dir)


if __name__ == "__main__":
    main()
