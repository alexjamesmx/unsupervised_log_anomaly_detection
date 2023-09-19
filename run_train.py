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


def build_vocab(vocab_path: str,
                data_dir: str,
                train_path: str,
                embeddings: str,
                embedding_dim: int = 300,
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
    if not os.path.exists(vocab_path):
        with open(train_path, 'rb') as f:
            data = pickle.load(f)
        logs = [x['EventTemplate']
                for x in data if np.max(x['Label']) == 0]
        logger.info(f"Lenght of logs eventTemplate: {len(logs)}")
        vocab = Vocab(logs, os.path.join(data_dir, embeddings),
                      embedding_dim=embedding_dim)
        logger.info(f"Save vocab in {vocab_path}")
        logger.info(f"Vocab size: {len(vocab)}")
        vocab.save_vocab(vocab_path)
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


def train(args: argparse.Namespace,
          train_path: str,
          test_path: str,
          vocab: Vocab,
          model: torch.nn.Module,
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

    logger.info(f"Main: log sequences statistics: {stat}")
    data = shuffle(data)
    logger.info(f"Main: log sequences length: {str(len(data))}")
    n_valid = int(len(data) * args.valid_ratio)
    train_data, valid_data = data[:-n_valid], data[-n_valid:]

    logger.info(
        f"Main: ,training, valid size: {len(train_data)}, {str(n_valid)} where valid ratio is {args.valid_ratio}")

    print("\nBuilding train dataset\n")
    sequentials, quantitatives, semantics, labels, idxs, _ = sliding_window(
        train_data,
        vocab=vocab,
        window_size=args.history_size,
        is_train=True,
        semantic=args.semantic,
        quantitative=args.quantitative,
        sequential=args.sequential,
        logger=logger
    )

    train_dataset = LogDataset(
        sequentials, quantitatives, semantics, labels, idxs)
    print("\nBuilding valid dataset\n")
    sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels = sliding_window(
        valid_data,
        vocab=vocab,
        window_size=args.history_size,
        is_train=False,
        semantic=args.semantic,
        quantitative=args.quantitative,
        sequential=args.sequential,
        logger=logger
    )
    valid_dataset = LogDataset(
        sequentials, quantitatives, semantics, labels, sequence_idxs)
    logger.info(
        f"Train dataset: {len(train_dataset)}, Valid dataset: {len(valid_dataset)}")
    optimizer = get_optimizer(args, model.parameters())
    device = accelerator.device
    model = model.to(device)

    logger.info(f"Start training {args.model_name} model on {device} device\n")
    logger.info(model)

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

    train_loss, val_loss, val_acc, args.topk = trainer.train(device=device,
                                                             save_dir=f"{args.output_dir}/models",
                                                             model_name=args.model_name,
                                                             topk=args.topk)
    logger.info(
        f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    logger.info(
        f"Length of sessions labels: {len(session_labels)}, Length of valid dataset: {len(valid_dataset)}")
    acc, recommend_topk = trainer.predict_unsupervised(valid_dataset,
                                                       session_labels,
                                                       topk=args.topk,
                                                       device=device,
                                                       is_valid=True)
    logger.info(
        f"Validation Result:: Acc: {acc:.4f}, Top-{args.topk} Recommendation: {recommend_topk}\n")
    # now load test data
    print("Loading test dataset")
    data, stat = load_features(test_path,
                               is_train=False)
    logger.info(f"Test data statistics: {stat}")
    label_dict = {}
    counter = {}
    for (e, s, l) in data:
        label_dict[tuple(s)] = l
        try:
            counter[tuple(s)] += 1
        except Exception:
            counter[tuple(s)] = 1

    # Label dict e.g {('Receiving block <*> src: /<*> dest: /<*>', 'Receiving block <*> src: /<*> dest: /<*>', 'Receiving block <*> src: /<*> dest: /<*>', 'BLOCK* NameSystem.allocateBlock: <*> <*>', 'PacketResponder <*> for block <*> <*>', 'Received block <*> of size <*> from /<*>', 'BLOCK* NameSystem.addStoredBlock: blockMap updated: <*> is added to <*> size <*>', 'PacketResponder <*> for block <*> <*>', 'PacketResponder <*> for block <*> <*>', 'Received block <*> of size <*> from /<*>', 'Received block <*> of size <*> from /<*>', 'BLOCK* NameSystem.addStoredBlock: blockMap updated: <*> is added to <*> size <*>', 'BLOCK* NameSystem.addStoredBlock: blockMap updated: <*> is added to <*> size <*>', 'Verification succeeded for <*>', 'Verification succeeded for <*>', 'BLOCK* NameSystem.delete: <*> is added to invalidSet of <*>', 'BLOCK* NameSystem.delete: <*> is added to invalidSet of <*>', 'BLOCK* NameSystem.delete: <*> is added to invalidSet of <*>', 'Deleting block <*> file <*>', 'Deleting block <*> file <*>', 'Deleting block <*> file <*>'): 0,...}
    data = [(list(k), v) for k, v in label_dict.items()]
    # num_sessions = [n,n,n,n] where n is the number of times a session appears in the test data
    num_sessions = [counter[tuple(k)] for k, _ in data]

    sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels = sliding_window(
        data,
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
    logger.info(f"Test dataset: {len(test_dataset)}")
    logger.info(
        f"Start predicting {args.model_name} model on {device} device with top-{args.topk} recommendation")
    acc, f1, pre, rec = trainer.predict_unsupervised(test_dataset,
                                                     session_labels,
                                                     topk=args.topk,
                                                     device=device,
                                                     is_valid=False,
                                                     num_sessions=num_sessions,)

    logger.info(
        f"Test Result:: Acc: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    logger.info(
        f"window size: {args.window_size}, history size: {args.history_size}")
    return acc, f1, pre, rec


def run_train(args, accelerator, logger):

    if args.grouping == "sliding":
        args.output_dir = f"{args.output_dir}{args.dataset_name}/sliding/W{args.window_size}_S{args.step_size}_C{args.is_chronological}_train{args.train_size}"

    else:
        args.output_dir = f"{args.output_dir}{args.dataset_name}/session/train{args.train_size}"

    train_path, test_path = process_dataset(logger, data_dir=args.data_dir, output_dir=args.output_dir,
                                            log_file=args.log_file,
                                            dataset_name=args.dataset_name, grouping=args.grouping,
                                            window_size=args.window_size, step_size=args.step_size,
                                            train_size=args.train_size, is_chronological=args.is_chronological,
                                            session_type=args.session_level)
    os.makedirs(f"{args.output_dir}/vocabs", exist_ok=True)
    vocab_path = f"{args.output_dir}/vocabs/{args.model_name}.pkl"
    log_vocab = build_vocab(vocab_path,
                            args.data_dir,
                            train_path,
                            args.embeddings,
                            embedding_dim=args.embedding_dim,
                            logger=logger)
    model = build_model(args, vocab_size=len(log_vocab))
    train(args,
          train_path,
          test_path,
          log_vocab,
          model,
          logger=logger,
          accelerator=accelerator)
