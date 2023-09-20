from collections import Counter
import pickle
from tqdm import tqdm
from typing import List, Tuple, Optional, Any
import numpy as np
import sys
from logadempirical.data.log import Log


def load_features(data_path, min_len=0, is_train=True, log=Log):
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
                    (seq["SessionId"], seq['EventTemplate'], label))
            else:
                no_abnormal += 1
        print("Number of abnormal sessions:", no_abnormal, "")
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
            logs.append((seq["SessionId"], seq['EventTemplate'], label))
        print("Number of abnormal sessions:", no_abnormal)

    # length is the length of each log sequence (window type) at position 0 (list of events), where each log is an array of [eventTemplate, label].
    # In a nutshell, it is the length of the log sequence

    logs_len = [len(log[1]) for log in logs]
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

    if is_train:

        for idx, (*eventId, templates, labels) in tqdm(enumerate(data), total=len(data),
                                                       desc=f"Sliding window with size {window_size}"):

            line = list(templates)
            line = line + [vocab.pad_token] * (window_size - len(line) + 1)
            if isinstance(labels, list):
                labels = [0] * (window_size - len(labels) + 1) + labels
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
    else:
        eventIds = {}
        for idx, (templates, [eventId, labels]) in tqdm(enumerate(data), total=len(data),
                                                        desc=f"Sliding window with size {window_size}"):

            line = list(templates)
            line = line + [vocab.pad_token] * (window_size - len(line) + 1)
            if isinstance(labels, list):
                labels = [0] * (window_size - len(labels) + 1) + labels
            session_labels[idx] = labels if isinstance(
                labels, int) else max(labels)

            if eventId:
                eventIds[idx] = (eventId)
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

        return sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels, eventIds
