import os
import pandas as pd
from logadempirical.data.grouping import time_sliding_window, session_window, session_window_bgl, fixed_window
import pickle
from sklearn.utils import shuffle
from logging import Logger
from typing import List, Tuple
import sys
from logadempirical.data.log import Log


def process_dataset(logger: Logger,
                    data_dir: str,
                    output_dir: str,
                    log_file: str,
                    dataset_name: str,
                    grouping: str,
                    window_size: int,
                    step_size: int,
                    train_size: float,
                    is_chronological: bool = False,
                    session_type: str = "entry",
                    storeLog: Log = None
                    ) -> Tuple[str, str]:
    """
    creating log sequences by sliding window
    :param logger:
    :param data_dir:
    :param output_dir:
    :param log_file:
    :param dataset_name:
    :param grouping:
    :param window_size:
    :param step_size:
    :param train_size:
    :param is_chronological:
    :param session_type:
    :return:
    """

    if os.path.exists(os.path.join(output_dir, "train.pkl")) and os.path.exists(os.path.join(output_dir, "test.pkl")):
        logger.info(
            f"Loading {output_dir}/train.pkl and {output_dir}/test.pkl")
        return os.path.join(output_dir, "train.pkl"), os.path.join(output_dir, "test.pkl")

    logger.info(
        f"Main: Loading new sequences {data_dir}/{log_file}_structured.csv")
    df = pd.read_csv(f'{data_dir}/{log_file}_structured.csv')

    logger.info(f"process dataset: Grouping: {grouping}")
    logger.info(f"process dataset: unlabeled size: {len(df)}")
    if grouping == "sliding":
        df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
        n_train = int(len(df) * train_size)
        if session_type == "entry":
            sliding = fixed_window
        elif session_type == "time":
            sliding = time_sliding_window
            window_size = window_size
            step_size = step_size
        if not is_chronological:
            window_df = sliding(
                df[["Label", "EventId", "EventTemplate", "Content"]],
                window_size=window_size,
                step_size=step_size
            )
            window_df = shuffle(window_df)
            n_train = int(len(window_df) * train_size)
            train_window = window_df[:n_train]
            test_window = window_df[n_train:]
        else:
            train_window = sliding(
                df[["Timestamp", "Label", "EventId",
                    "EventTemplate", "Content"]].iloc[:n_train, :],
                window_size=window_size,
                step_size=step_size
            )
            test_window = sliding(
                df[["Timestamp", "Label", "EventId", "EventTemplate", "Content"]].iloc[n_train:, :].reset_index(
                    drop=True),
                window_size=window_size,
                step_size=step_size
            )

    elif grouping == "session":
        if dataset_name == "HDFS":
            id_regex = r'(blk_-?\d+)'
            label_dict = {}
            blk_label_file = os.path.join(data_dir, "anomaly_label.csv")
            blk_df = pd.read_csv(blk_label_file)
            logger.info(f"process dataset: anomaly df size: {len(blk_df)}")
            # Anomaly labels to 1, Normal labels to 0
            for _, row in enumerate(blk_df.to_dict("records")):
                label_dict[row["BlockId"]
                           ] = 1 if row["Label"] == "Anomaly" else 0
            # use label_dict and df to create window_df
            window_df = session_window(
                df, id_regex, label_dict, window_size=int(window_size))
            n_train = int(len(window_df) * train_size)
            logger.info(
                f"process dataset: window_df (sessions) size {len(window_df)}, train_size {train_size}, n_train: {len(window_df) * train_size}")
            train_window = window_df[:n_train]
            test_window = window_df[n_train:]
        elif dataset_name == "BGL":
            # df["NodeId"] = df["Node"].apply(lambda x: str(x).split(":")[0])
            # window df
            window_df = session_window_bgl(df)
            # number of train, test samples
            n_train = int(len(window_df) * train_size)
            train_window = window_df[:n_train]
            test_window = window_df[n_train:]
        else:
            raise NotImplementedError(
                f"{dataset_name} with {grouping} is not implemented")
    else:
        raise NotImplementedError(f"{grouping} is not implemented")

    logger.info(
        f"Saving sessions at {output_dir}/train.pkl and {output_dir}/test.pkl\n")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), mode="wb") as f:
        pickle.dump(train_window, f)
    with open(os.path.join(output_dir, "test.pkl"), mode="wb") as f:
        pickle.dump(test_window, f)
    return os.path.join(output_dir, "train.pkl"), os.path.join(output_dir, "test.pkl")


if __name__ == '__main__':
    print("DATA LOADER ES MAIN")
    process_dataset(Logger("BGL"),
                    data_dir="../../dataset/", output_dir="../../dataset/", log_file="BGL.log",
                    dataset_name="bgl",
                    grouping="sliding", window_size=10, step_size=10, train_size=0.8, is_chronological=True,
                    session_type="entry")
