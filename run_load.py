import os
import pickle
from collections import Counter

import torch
import yaml
from sklearn.utils import shuffle

from accelerate import Accelerator
import logging
from logging import getLogger, Logger
import argparse
from typing import List, Tuple, Optional
import numpy as np

from logadempirical.data.data_loader import process_dataset
from logadempirical.data.vocab import Vocab
from logadempirical.data.feature_extraction import load_features, sliding_window
from logadempirical.data.dataset import LogDataset
from logadempirical.helpers import arg_parser, get_optimizer
from logadempirical.models import get_model, ModelConfig
from logadempirical.trainer import Trainer
from logadempirical.data.log import Log


def build_vocab(vocab_path: str,
                logger: Logger = getLogger("__name__")) -> Vocab:
    """
    Build vocab from training data
    Parameters
    ----------
    vocab_path: str: Path to save vocab
    data_dir: str: Path to data directory
    train_path: str: Path to training data
    embeddings: str: Path to pretrained embeddings
    embedding_dim: int: Dimension of embeddings
    logger: Logger: Logger

    Returns
    -------
    vocab: Vocab: Vocabulary
    """
    if os.path.exists(vocab_path):
        vocab = Vocab.load_vocab(vocab_path)
        logger.info(f"Load vocab from {vocab_path}")
        logger.info(f"Vocab size: {len(vocab)}")
    else:
        raise FileNotFoundError(f"Vocab file {vocab_path} not found")
    return vocab


def build_model(args, vocab_size):
    """
    Build model
    Parameters
    ----------
    args: argparse.Namespace: Arguments
    vocab_size: int: Size of vocabulary

    Returns
    -------

    """
    if args.model_name == "DeepLog":
        model_config = ModelConfig(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            criterion=torch.nn.CrossEntropyLoss(ignore_index=0)
        )
    elif args.model_name == "LogAnomaly":
        model_config = ModelConfig(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            criterion=torch.nn.CrossEntropyLoss(ignore_index=0),
            use_semantic=args.semantic
        )
    else:
        raise NotImplementedError
    model = get_model(args.model_name, model_config)
    return model


def train_and_eval(args: argparse.Namespace,
                   train_path: str,
                   test_path: str,
                   vocab: Vocab,
                   model: torch.nn.Module,
                   log: Log,
                   logger: Logger = getLogger("__name__"),
                   accelerator: Accelerator = Accelerator()
                   ) -> Tuple[float, float, float, float]:
    """
    Run model
    Parameters
    ----------
    args: argparse.Namespace: Arguments
    train_path: str: Path to training data
    test_path: str: Path to test data
    vocab: Vocab: Vocabulary
    model: torch.nn.Module: Model
    logger: Logger: Logger

    Returns
    -------
    Accuracy metrics
    """
    data, stat = load_features(train_path,
                               is_train=True)

    logger.info(f"Main: Train log sequences statistics: {stat}")
    data = shuffle(data)
    n_valid = int(len(data) * args.valid_ratio)
    train_data, valid_data = data[:-n_valid], data[-n_valid:]
    log.set_train_data(train_data)

    sequentials, quantitatives, semantics, labels, idxs, _ = sliding_window(
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
                                 semantics, labels, idxs)

    train_dataset = LogDataset(
        sequentials, quantitatives, semantics, labels, idxs)
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
        sequentials, quantitatives, semantics, labels, sequence_idxs)

    optimizer = get_optimizer(args, model.parameters())
    device = accelerator.device
    model = model.to(device)

    logger.info(f"Start training {args.model_name} model on {device} device")
    logger.info(f"{model}\n")

    trainer = Trainer(
        model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        is_train=True,
        optimizer=optimizer,
        no_epochs=args.max_epoch,
        batch_size=args.batch_size,
        scheduler_type=args.scheduler,
        warmup_rate=args.warmup_rate,
        accumulation_step=args.accumulation_step,
        logger=logger,
        accelerator=accelerator,
        num_classes=len(vocab),
    )
    logger.info(
        f"Loading model from {args.output_dir}/models/{args.model_name}.pt\n")

    trainer.load_model(f"{args.output_dir}/models/{args.model_name}.pt")

    acc, recommend_topk = trainer.predict_unsupervised(valid_dataset,
                                                       session_labels,
                                                       topk=args.topk,
                                                       device=device,
                                                       is_valid=True)
    logger.info(
        f"Validation Result:: Acc: {acc:.4f}, Top-{args.topk} Recommendation: {recommend_topk}")

    # now load test data
    test_data, stat = load_features(test_path,
                                    is_train=False)

    test_data = test_data[:1000]
    log.set_test_data(test_data)
    train_loss = None
    if args.update:
        print("Updating model")
        false_positive_data = log.get_test_data(
            blockId="blk_-41265708926987771")
        print(false_positive_data)
        if len(false_positive_data) == 0:
            raise Exception("False positive with id = n is not found")
        sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels = sliding_window(
            false_positive_data,
            vocab=vocab,
            window_size=args.history_size,
            is_train=True,
            semantic=args.semantic,
            quantitative=args.quantitative,
            logger=logger
        )
        false_positive_dataset = LogDataset(
            sequentials, quantitatives, semantics, labels, sequence_idxs)
        train_loss, args.topk = trainer.train_on_false_positive(false_positive_dataset=false_positive_dataset,
                                                                device=device,
                                                                save_dir=f"{args.output_dir}/models",
                                                                model_name=args.model_name,
                                                                topk=args.topk)
        logger.info(f"UPDATED MODEL Train Loss: {train_loss:.4f}")

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
    test_data = [(list(k), v) for k, v in test_data if v[1] == 1]
    print("REAL ANOMALIES HERE: ", len(test_data))
    num_sessions = [counter[tuple(k)] for k, _ in test_data]
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

    log.set_valid_data(valid_data)
    log.set_lengths(len(train_data), len(valid_data), len(test_data))

    log.set_test_sliding_window(sequentials, quantitatives,
                                semantics, labels, sequence_idxs, session_labels, eventIds)
    log.get_lenths()
    log.get_train_sliding_window(length=True)
    log.get_valid_sliding_window(length=True)
    log.get_test_sliding_window(length=True)

    test_dataset = LogDataset(
        sequentials, quantitatives, semantics, labels, sequence_idxs)

    # START PREDICTING
    logger.info(
        f"Start predicting {args.model_name} model on {device} device with top-{args.topk} recommendation")
    acc, f1, pre, rec = trainer.predict_unsupervised(test_dataset,
                                                     session_labels,
                                                     eventIds=eventIds,
                                                     topk=args.topk,
                                                     device=device,
                                                     is_valid=False,
                                                     num_sessions=num_sessions,
                                                     log=log
                                                     )

    logger.info(
        f"Test Result:: Acc: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return acc, f1, pre, rec


def run_load(args, accelerator, logger):
    if args.grouping == "sliding":
        args.output_dir = f"{args.output_dir}{args.dataset_name}/sliding/W{args.window_size}_S{args.step_size}_C{args.is_chronological}_train{args.train_size}"

    else:
        args.output_dir = f"{args.output_dir}{args.dataset_name}/session/train{args.train_size}"

    log = Log()

    train_path, test_path = process_dataset(logger, data_dir=args.data_dir, output_dir=args.output_dir,
                                            log_file=args.log_file,
                                            dataset_name=args.dataset_name, grouping=args.grouping,
                                            window_size=args.window_size, step_size=args.step_size,
                                            train_size=args.train_size, is_chronological=args.is_chronological,
                                            session_type=args.session_level,
                                            log=log)

    os.makedirs(f"{args.output_dir}/vocabs", exist_ok=True)
    vocab_path = f"{args.output_dir}/vocabs/{args.model_name}.pkl"

    log_vocab = build_vocab(vocab_path, logger=logger)

    model = build_model(args, vocab_size=len(log_vocab))
    train_and_eval(args,
                   train_path,
                   test_path,
                   log_vocab,
                   model,
                   log,
                   logger=logger,
                   accelerator=accelerator)
