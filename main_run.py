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

from logadempirical.data import process_dataset
from logadempirical.data.vocab import Vocab
from logadempirical.data.feature_extraction import load_features, sliding_window
from logadempirical.data.dataset import LogDataset
from logadempirical.helpers import arg_parser, get_optimizer
from logadempirical.models import get_model, ModelConfig
from logadempirical.trainer import Trainer

import sys

# Logging config
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

accelerator = Accelerator()


def build_vocab(vocab_path: str,
                data_dir: str,
                train_path: str,
                embeddings: str,
                embedding_dim: int = 300,
                is_unsupervised: bool = False,
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
    is_unsupervised: bool: Whether the model is unsupervised or not
    logger: Logger: Logger

    Returns
    -------
    vocab: Vocab: Vocabulary
    """
    # save vocab if it doesn't exist
    if not os.path.exists(vocab_path):
        with open(train_path, 'rb') as f:
            data = pickle.load(f)
        # NOTE - vocab is created with the EventTemplates
        if is_unsupervised:
            logs = [x['EventTemplate']
                    for x in data if np.max(x['Label']) == 0]

        logger.info(f"Lenght of logs eventTemplate: {len(logs)}")

        vocab = Vocab(logs, os.path.join(data_dir, embeddings),
                      embedding_dim=embedding_dim)
        logger.info(f"Save vocab in {vocab_path}")
        logger.info(f"Vocab size: {len(vocab)}")
        vocab.save_vocab(vocab_path)
    # load vocab if it exists
    else:
        vocab = Vocab.load_vocab(vocab_path)
        logger.info(f"Load vocab from {vocab_path}")
        logger.info(f"Vocab size: {len(vocab)}")

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
                   is_unsupervised=False,
                   logger: Logger = getLogger("__name__")) -> Tuple[float, float, float, float]:
    """
    Run model
    Parameters
    ----------
    args: argparse.Namespace: Arguments
    train_path: str: Path to training data
    test_path: str: Path to test data
    vocab: Vocab: Vocabulary
    model: torch.nn.Module: Model
    is_unsupervised: bool: Whether the model is unsupervised or not
    logger: Logger: Logger

    Returns
    -------
    Accuracy metrics
    """
    logger.info("Loading train and eval...")
    logger.info(f"train path: {train_path}")
    data, stat = load_features(train_path,
                               is_unsupervised=is_unsupervised,
                               is_train=True)

    logger.info(f"Main: log sequences statistics: {stat}")
    data = shuffle(data)
    logger.info(f"Main: log sequences length: {str(len(data))}")
    n_valid = int(len(data) * args.valid_ratio)
    train_data, valid_data = data[:-n_valid], data[-n_valid:]

    logger.info(
        f"Main: ,training, valid size: {len(train_data)}, {str(n_valid)} where valid ratio is {args.valid_ratio}")

    # NOTE - copy the train data to a file
    output_data_first = [
        f"Index {i}: {log}" for i, log in enumerate(train_data[:10], start=1)]

    output_data_last = [
        f"Index {i}: {log}" for i, log in enumerate(train_data[len(train_data)-10:], start=len(train_data)-9)]

    with open("./testing/sessions_training_before_sliding_windows.txt", "w") as f:
        sys.stdout = f
        for item in output_data_first:
            f.write("%s\n" % item)
        for item in output_data_last:
            f.write("%s\n" % item)
    sys.stdout = sys.__stdout__
    ################
    # build train dataset
    sequentials, quantitatives, semantics, labels, idxs, _ = sliding_window(
        train_data,
        vocab=vocab,
        window_size=args.history_size,
        is_train=True,
        semantic=args.semantic,
        quantitative=args.quantitative,
        sequential=args.sequential,
        is_unsupervised=is_unsupervised,
        logger=logger
    )

    train_dataset = LogDataset(
        sequentials, quantitatives, semantics, labels, idxs)
    # build valid dataset
    sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels = sliding_window(
        valid_data,
        vocab=vocab,
        window_size=args.history_size,
        is_train=False,
        semantic=args.semantic,
        quantitative=args.quantitative,
        sequential=args.sequential,
        is_unsupervised=is_unsupervised,
        logger=logger
    )
    valid_dataset = LogDataset(
        sequentials, quantitatives, semantics, labels, sequence_idxs)
    logger.info(f"Train dataset: {len(train_dataset)}")
    logger.info(f"Valid dataset: {len(valid_dataset)}")
    optimizer = get_optimizer(args, model.parameters())

    device = accelerator.device
    model = model.to(device)

    # start training
    logger.info(f"Start training {args.model_name} model on {device} device")

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
        num_classes=len(vocab) if is_unsupervised else args.n_class,
    )
    # load model if resuming
    if args.resume and os.path.exists(f"{args.output_dir}/models/{args.model_name}.pt"):
        logger.info(
            f"Loading model from {args.output_dir}/models/{args.model_name}.pt...")
        trainer.load_model(f"{args.output_dir}/models/{args.model_name}.pt")

    # save model if training
    if args.train:
        train_loss, val_loss, val_acc = trainer.train(device=device,
                                                      save_dir=f"{args.output_dir}/models",
                                                      model_name=args.model_name,
                                                      topk=1 if not is_unsupervised else args.topk)
        logger.info(
            f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    if is_unsupervised:
        acc, recommend_topk = trainer.predict_unsupervised(valid_dataset,
                                                           session_labels,
                                                           topk=args.topk,
                                                           device=device,
                                                           is_valid=True)
        logger.info(
            f"Validation Result:: Acc: {acc:.4f}, Top-{args.topk} Recommendation: {recommend_topk}")
    else:
        acc, f1, pre, rec = trainer.predict_supervised(valid_dataset,
                                                       session_labels,
                                                       device=device)
        logger.info(
            f"Validation Result:: Acc: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    # now load test data
    print("Loading test dataset\n")
    data, stat = load_features(test_path,
                               is_unsupervised=is_unsupervised,
                               is_train=False)
    logger.info(f"Test data statistics: {stat}")
    label_dict = {}
    counter = {}
    for (s, l) in data:
        label_dict[tuple(s)] = l
        try:
            counter[tuple(s)] += 1
        except Exception:
            counter[tuple(s)] = 1
    data = [(list(k), v) for k, v in label_dict.items()]

    num_sessions = [counter[tuple(k)] for k, _ in data]
    sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels = sliding_window(
        data,
        vocab=vocab,
        window_size=args.history_size,
        is_train=False,
        semantic=args.semantic,
        quantitative=args.quantitative,
        sequential=args.sequential,
        is_unsupervised=is_unsupervised,
        logger=logger
    )

    test_dataset = LogDataset(
        sequentials, quantitatives, semantics, labels, sequence_idxs)
    logger.info(f"Test dataset: {len(test_dataset)}")
    if is_unsupervised:
        acc, f1, pre, rec = trainer.predict_unsupervised(test_dataset,
                                                         session_labels,
                                                         topk=args.topk,
                                                         device=device,
                                                         is_valid=False,
                                                         num_sessions=num_sessions)
    else:
        acc, f1, pre, rec = trainer.predict_supervised(test_dataset,
                                                       session_labels,
                                                       device=device)
    logger.info(
        f"Test Result:: Acc: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return acc, f1, pre, rec


def run(args):
    # logger initialization
    logger = getLogger(args.model_name)
    logger.info(accelerator.state)
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # output paths for train and test data
    if args.grouping == "sliding":
        args.output_dir = f"{args.output_dir}{args.dataset_name}/sliding/W{args.window_size}_S{args.step_size}_C{args.is_chronological}_train{args.train_size}"

    else:
        args.output_dir = f"{args.output_dir}{args.dataset_name}/session/train{args.train_size}"

    logger.info(f"Main: Output directory: {args.output_dir}")
    # proccess dataset
    train_path, test_path = process_dataset(logger, data_dir=args.data_dir, output_dir=args.output_dir,
                                            log_file=args.log_file,
                                            dataset_name=args.dataset_name, grouping=args.grouping,
                                            window_size=args.window_size, step_size=args.step_size,
                                            train_size=args.train_size, is_chronological=args.is_chronological,
                                            session_type=args.session_level)

    # create output directories for vocabs
    os.makedirs(f"{args.output_dir}/vocabs", exist_ok=True)
    vocab_path = f"{args.output_dir}/vocabs/{args.model_name}.pkl"
    # is model unsupervised
    is_unsupervised = args.model_name in ["LogAnomaly", "DeepLog"]
    # build vocab
    log_vocab = build_vocab(vocab_path,
                            args.data_dir,
                            train_path,
                            args.embeddings,
                            embedding_dim=args.embedding_dim,
                            is_unsupervised=is_unsupervised,
                            logger=logger)
    # NOTE am here build model, and embeddings yet to analyze

    model = build_model(args, vocab_size=len(log_vocab))
    # train and evaluate model
    train_and_eval(args,
                   train_path,
                   test_path,
                   log_vocab,
                   model,
                   is_unsupervised=is_unsupervised,
                   logger=logger)


# When run program
if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()

    if args.config_file is not None and os.path.exists(args.config_file):
        config_file = args.config_file
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            args = argparse.Namespace(**config)
        print(f"Loaded config from {config_file}!")
    run(args)
