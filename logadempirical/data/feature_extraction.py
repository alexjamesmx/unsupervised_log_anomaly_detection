from collections import Counter
import pickle
from tqdm import tqdm
from typing import List, Tuple, Optional, Any
import numpy as np
import sys


def load_features(data_path, is_unsupervised=True, min_len=0, is_train=True):
    """
    Load features from pickle file
    Parameters
    ----------
    data_path: str: Path to pickle file
    is_unsupervised: bool: Whether the model is unsupervised or not
    min_len: int: Minimum length of log sequence
    pad_token: str: Padding token
    is_train: bool: Whether the data is training data or not
    Returns
    -------
    logs: List[Tuple[List[str], int]]: List of log sequences
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # NOTE - copy the train data to a file
    output_data_first_10 = [
        f"Training data {i}: {log}" for i, log in enumerate(data[:10], start=1)]

    output_data_last_10 = [
        f"Training data {i}: {log}" for i, log in enumerate(data[len(data)-10:], start=len(data)-9)]

    with open("./testing/train_data_before_load_features.txt", "w") as f:
        sys.stdout = f
        print("Training data before load features: \n")
        for item in output_data_first_10:
            f.write("%s\n" % item)
        for item in output_data_last_10:
            f.write("%s\n" % item)
    sys.stdout = sys.__stdout__
    ###############
    print(f"\nLoading features...")

    print(f"load features: train length: {len(data)}")
    # for i, log in enumerate(data, start=1):
    # print(f"load features: train data {i}: {log}")
    if is_train:
        if is_unsupervised:
            logs = []
            no_abnormal = 0
            # print(f"feature extraction: first log sequence {data[0]}")
            for seq in data:
                # NOTE only true when min_len > 0
                if len(seq['EventTemplate']) < min_len:
                    print(
                        f"Feature extraction: \n len(seq['EventTemplate']) < min_len: {len(seq['EventTemplate'])}")
                    continue
                if not isinstance(seq['Label'], int):
                    print(
                        f"Feature extraction: \n NOT instance: seq['Label']: {seq['Label']}")
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
    logs_len = [len(log[0]) for log in logs]
    return logs, {"min": min(logs_len), "max": max(logs_len), "mean": np.mean(logs_len)}


def sliding_window(data: List[Tuple[List[str], int]],
                   window_size: int = 40,
                   is_train: bool = True,
                   vocab: Optional[Any] = None,
                   is_unsupervised: bool = True,
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
    is_unsupervised: bool: Whether the model is unsupervised or not
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
    print("\nSliding windows where window size is ", window_size)

    # for each log
    for idx, (templates, labels) in tqdm(enumerate(data), total=len(data),
                                         desc=f"Sliding window with size {window_size}"):
        line = list(templates)
        # add n padding tokens to the end of the sequence if window size is different from the length of the sequence
        line = line + [vocab.pad_token] * \
            (window_size - len(line) + is_unsupervised)
        if isinstance(labels, list):
            labels = [0] * (window_size - len(labels) +
                            is_unsupervised) + labels
        session_labels[idx] = labels if isinstance(
            labels, int) else max(labels)
        # for all events in a log
        for i in range(len(line) - window_size if is_unsupervised else len(line) - window_size + 1):
            if is_unsupervised:
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

            sequence = {'sequential': sequential_pattern}
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
    logger.info(f"Number of sequences: {len(labels)}")
    if not is_unsupervised:
        logger.info(f"Number of normal sequence: {len(labels) - sum(labels)}")
        logger.info(f"Number of abnormal sequence: {sum(labels)}")

    logger.warning(
        f"Number of unique abnormal events: {len(unique_ab_events)}")
    logger.info(
        f"Number of abnormal sessions: {sum(session_labels.values())}/{len(session_labels)}")

    return sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels
