from collections import Counter
import pickle
from tqdm import tqdm
from typing import List, Tuple, Optional, Any
import numpy as np
import sys


def load_features(data_path, min_len=0, is_train=True):
    """
    Load features from pickle file
    Parameters
    ----------
    data_path: str: Path to pickle file
    min_len: int: Minimum length of log sequence
    pad_token: str: Padding token
    is_train: bool: Whether the data is training data or not
    Returns
    -------
    logs: List[Tuple[List[str], int]]: List of log sequences
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print(f"\nLoading features...")

    print(f"load features: train length: {len(data)}")

    if is_train:
        logs = []
        no_abnormal = 0
        # print(f"feature extraction: first log sequence {data[0]}")
        for seq in data:
            # NOTE only true when min_len > 0
            if len(seq['EventTemplate']) < min_len:
                continue
            if not isinstance(seq['Label'], int):
                label = max(seq['Label'])
            else:
                label = seq['Label']
            if label == 0:
                logs.append(
                    (seq['EventTemplate'], label))
            else:
                no_abnormal += 1
        print("Number of abnormal sessions:", no_abnormal, "\n")
    else:
        logs = []
        no_abnormal = 0
        for seq in data:
            if len(seq['EventTemplate']) < min_len:
                continue
            if not isinstance(seq['Label'], int):
                label = seq['Label']
                if max(label) > 0:
                    no_abnormal += 1
            else:
                label = seq['Label']
                if label > 0:
                    no_abnormal += 1
            logs.append((seq['EventTemplate'], label))
        print("Number of abnormal sessions:", no_abnormal)

    # NOTE length is the length of each log sequence (window type) at position 0 (list of events), where each log is an array of [eventTemplate, label].
    # In a nutshell, it is the length of the log sequence
    if is_train:
      # NOTE - copy the train data to a file
        output_data_first_10 = [
            f"{i}: {log}" for i, log in enumerate(logs[:10], start=1)]

        output_data_last_10 = [
            f"{i}: {log}" for i, log in enumerate(logs[len(logs)-10:], start=len(logs)-9)]

        with open("./testing/3_train_data_after_load_features.txt", "w") as f:
            sys.stdout = f
            print("Training data after load features: \nOnly normal labels\n")
            for item in output_data_first_10:
                f.write("%s\n" % item)
            f.write("\n")
            for item in output_data_last_10:
                f.write("%s\n" % item)
            logs_len = [len(log[0]) for log in logs]
            f.write(
                f"max: {max(logs_len)} min: {min(logs_len)} mean: {np.mean(logs_len)}\n")
        sys.stdout = sys.__stdout__
    logs_len = [len(log[0]) for log in logs]
    return logs, {"min": min(logs_len), "max": max(logs_len), "mean": np.mean(logs_len)}


def sliding_window(data: List[Tuple[List[str], int]],
                   window_size: int = 40,
                   is_train: bool = True,
                   vocab: Optional[Any] = None,
                   sequential: bool = False,
                   quantitative: bool = False,
                   semantic: bool = False,
                   logger: Optional[Any] = None,
                   ) -> Any:
    """
    Sliding window for log sequence
    Parameters
    ----------
    data: List[Tuple[List[str], int]]: List of log sequences
    window_size: int: Size of sliding window
    is_train: bool: training mode or not
    vocab: Optional[Any]: Vocabulary
    sequential: bool: Whether to use sequential features
    quantitative: bool: Whether to use quantitative features
    semantic: bool: Whether to use semantic features
    logger: Optional[Any]: Logger

    Returns
    -------
    lists of sequential, quantitative, semantic features, and labels
    """
    log_sequences = []
    session_labels = {}
    unique_ab_events = set()
    # NOTE here "is_train" is for analysis purpose, where files are copied to a file, and print statements are used. is the same as else conditional,
    # so if you want clean code you can remove the if else and just use the what else conditional contains, because test and train use the same code.

    if is_train:
        # NOTE - copy the train data to a file
        print("Creating vocab_stoi.txt")
        with open("./testing/vocab_stoi.txt", "w") as f:
            sys.stdout = f
            for k, v in vocab.stoi.items():
                print(k, v)
        sys.stdout = sys.__stdout__
        with open("./testing/5_sequential_quantitives_semantics.txt", "w") as f:
            # for each log
            for idx, (templates, labels) in tqdm(enumerate(data), total=len(data),
                                                 desc=f"Sliding window with size {window_size}"):
                line = list(templates)
                # add n padding tokens to the end of the sequence if window size is different from the length of the sequence
                line = line + [vocab.pad_token] * \
                    (window_size - len(line) + 1)

                # NOTE test padding
                # if len(line) != len(list(templates)):
                #     print("line length != list(templates) length ", line)

                if isinstance(labels, list):
                    print("labels is a list")
                    labels = [0] * (window_size - len(labels) +
                                    1) + labels

                session_labels[idx] = labels if isinstance(
                    labels, int) else max(labels)
                # NOTE testing - print the first session and slide window
                # if idx == 0:
                #     print(
                #         f"First session: length {len(data[0])} content {data[idx]} \n")
                # for all events in a log
                # NOTE - copy to a file
                # Check if the current iteration index is within the first 10 or last 10
                if idx < 10 or idx >= len(data) - 10:
                    output_line = f"Length: {len(line)}, Line: {line}"
                    output_labels = f"labels: {labels}"
                    output_session_labels = f"Session labels: {session_labels}"

                    # Write the information to the file
                    f.write(f"Iter {idx}\n")
                    f.write(f"{output_line}\n")
                    f.write(f"{output_labels}\n")
                    f.write(f"{output_session_labels}\n")
                    f.write("\n")
                ###############

                for i in range(len(line) - window_size):
                    # get the index of the event in the window sequence
                    label = vocab.get_event(line[i + window_size],
                                            use_similar=quantitative)  # use_similar only for LogAnomaly

                    seq = line[i: i + window_size]
                    # NOTE testing
                    # if i < 2 and idx < 1:
                    # print(f"sliding window: seq: {seq}")
                    # Get sequential patterns for all sequences in each log
                    sequential_pattern = [vocab.get_event(
                        event, use_similar=quantitative) for event in seq]
                    semantic_pattern = None
                    # print(f"sequential_pattern " + str(sequential_pattern) + "idx ", idx)
                    if semantic:
                        print("semantic")
                        semantic_pattern = [
                            vocab.get_embedding(event) for event in seq]
                    quantitative_pattern = None
                    if quantitative:
                        print("quantitive")
                        quantitative_pattern = [0] * len(vocab)
                        log_counter = Counter(sequential_pattern)
                        for key in log_counter:
                            try:
                                quantitative_pattern[key] = log_counter[key]
                            except Exception as _:
                                pass  # ignore unseen events or padding key

                    sequence = {'nSec': i, 'sequential': sequential_pattern}
                    if quantitative:
                        sequence['quantitative'] = quantitative_pattern
                    if semantic:
                        sequence['semantic'] = semantic_pattern
                    sequence['label'] = label
                    sequence['idx'] = idx
                    # NOTE testing
                    # print(f"sequence: {sequence}")
                    # if idx == len(data) - 1 and i == len(line) - window_size - 1:
                    #     print(f"sequence of last session: {sequence}", "\n")
                    #     print("vocab ", vocab.stoi, "\n")
                    #     print("last session: length", len(
                    #         data[-1][0]), "content", data[-1][0],  "\n")
                    #     print("index and line length ", idx, len(line), "\n")
                    #     print("label vocab get event ",
                    #           vocab.get_event(line[i + window_size]), "\n")
                    log_sequences.append(sequence)
            sequentials, quantitatives, semantics = None, None, None
            if sequential:
                print("sequential \n")
                sequentials = [seq['sequential'] for seq in log_sequences]
            if quantitative:
                print("quantitative \n")
                quantitatives = [seq['quantitative'] for seq in log_sequences]
            if semantic:
                print("semantic \n")
                semantics = [seq['semantic'] for seq in log_sequences]
            labels = [seq['label'] for seq in log_sequences]
            sequence_idxs = [seq['idx'] for seq in log_sequences]
            first_idx = [seq for seq in log_sequences if seq['idx'] == 0]
            last_idx = [
                seq for seq in log_sequences if seq['idx'] == len(data) - 1]

            f.write(
                f"Log sequences for first idx: {first_idx}\n\n")
            f.write(
                f"Log sequences for last idx: {last_idx}\n\n")

            logger.info(f"Number of sequences: {len(labels)}")

            # ? dont make sense to me
            # logger.warning(
            #     f"Number of unique abnormal events: {len(unique_ab_events)}")

            logger.info(
                f"Number of abnormal sessions: {sum(session_labels.values())}/{len(session_labels)}")
        sys.stdout = sys.__stdout__

        return sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels

    else:
        for idx, (templates, labels) in tqdm(enumerate(data), total=len(data),
                                             desc=f"Sliding window with size {window_size}"):
            line = list(templates)
            line = line + [vocab.pad_token] * \
                (window_size - len(line) + 1)
            if isinstance(labels, list):
                labels = [0] * (window_size - len(labels) +
                                1) + labels
            session_labels[idx] = labels if isinstance(
                labels, int) else max(labels)
            for i in range(len(line) - window_size):
                label = vocab.get_event(line[i + window_size],
                                        use_similar=quantitative)  # use_similar only for LogAnomaly

                seq = line[i: i + window_size]
                sequential_pattern = [vocab.get_event(
                    event, use_similar=quantitative) for event in seq]
                semantic_pattern = None
                if semantic:
                    semantic_pattern = [
                        vocab.get_embedding(event) for event in seq]
                quantitative_pattern = None
                if quantitative:
                    quantitative_pattern = [0] * len(vocab)
                    log_counter = Counter(sequential_pattern)
                    for key in log_counter:
                        try:
                            quantitative_pattern[key] = log_counter[key]
                        except Exception as _:
                            pass  # ignore unseen events or padding key

                sequence = {'sequential': sequential_pattern}
                if quantitative:
                    sequence['quantitative'] = quantitative_pattern
                if semantic:
                    sequence['semantic'] = semantic_pattern
                sequence['label'] = label
                sequence['idx'] = idx
                log_sequences.append(sequence)

        sequentials, quantitatives, semantics = None, None, None
        if sequential:
            sequentials = [seq['sequential'] for seq in log_sequences]
        if quantitative:
            quantitatives = [seq['quantitative'] for seq in log_sequences]
        if semantic:
            semantics = [seq['semantic'] for seq in log_sequences]
        labels = [seq['label'] for seq in log_sequences]
        sequence_idxs = [seq['idx'] for seq in log_sequences]
        logger.info(f"Number of sequences: {len(labels)}")

        logger.warning(
            f"Number of unique abnormal events: {len(unique_ab_events)}")
        logger.info(
            f"Number of abnormal sessions: {sum(session_labels.values())}/{len(session_labels)}\n")

        return sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels
