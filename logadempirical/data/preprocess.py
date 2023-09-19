import torch
from logadempirical.data.vocab import Vocab
from typing import Tuple
from logadempirical.data.feature_extraction import load_features, sliding_window
from logadempirical.data.log import Log
from sklearn.utils import shuffle
from logadempirical.data.dataset import LogDataset


def preprocess_data(path: str,
                    args,
                    is_train: bool,
                    log: Log,
                    logger) -> Tuple[list, list] or list:
    data, stat = load_features(path, is_train=is_train)
    phase_message = "train" if is_train else "test"
    logger.info(f"{phase_message} log sequences statistics: {stat}\n")

    if is_train:
        data = shuffle(data)
        n_valid = int(len(data) * args.valid_ratio)
        train_data, valid_data = data[:-n_valid], data[-n_valid:]
        log.set_train_data(train_data)
        log.set_valid_data(valid_data)
        log.set_lengths(train_length=len(train_data), valid_length=len(
            valid_data))
        return train_data, valid_data
    else:
        test_data = data[:500]
        log.set_test_data(test_data)
        log.set_lengths(test_length=len(test_data))
        label_dict = {}
        counter = {}
        for (e, s, l) in test_data:
            key = tuple(s)
            label_dict[key] = [e, l]
            try:
                counter[key] += 1
            except Exception:
                counter[key] = 1
        test_data = [(list(k), v) for k, v in label_dict.items()]
        # test_data = [(list(k), v) for k, v in test_data if v[1] == 1]
        print("REAL ANOMALIES HERE: ", len(test_data))
        num_sessions = [counter[tuple(k)] for k, _ in test_data]
        return test_data, num_sessions


def preprocess_slidings(train_data=None, valid_data=None, test_data=None,
                        vocab=Vocab, args=None,
                        is_train=bool,
                        log=Log,
                        logger=None):

    if is_train:
        sequentials, quantitatives, semantics, labels, sequence_idxs, _ = sliding_window(
            train_data,
            vocab=vocab,
            window_size=args.history_size,
            is_train=True,
            semantic=args.semantic,
            quantitative=args.quantitative,
            sequential=args.sequential,
            logger=logger,
        )
        log.set_train_sliding_window(sequentials, quantitatives,
                                     semantics, labels, sequence_idxs)

        train_dataset = LogDataset(
            sequentials, quantitatives, semantics, labels, sequence_idxs)
        sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels = sliding_window(
            valid_data,
            vocab=vocab,
            window_size=args.history_size,
            is_train=True,
            semantic=args.semantic,
            quantitative=args.quantitative,
            sequential=args.sequential,
            logger=logger
        )
        log.set_valid_sliding_window(sequentials, quantitatives,
                                     semantics, labels, sequence_idxs, session_labels)
        valid_dataset = LogDataset(
            sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels=session_labels)
        return train_dataset, valid_dataset
    else:
        sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels, eventIds = sliding_window(
            test_data,
            vocab=vocab,
            window_size=args.history_size,
            is_train=False,
            semantic=args.semantic,
            quantitative=args.quantitative,
            sequential=args.sequential,
            logger=logger
        )
        test_dataset = LogDataset(
            sequentials, quantitatives, semantics, labels, sequence_idxs)
        log.set_test_sliding_window(sequentials, quantitatives,
                                    semantics, labels, sequence_idxs, session_labels, eventIds)
        return test_dataset, eventIds
