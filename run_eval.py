import os
import pickle
import torch
import argparse
import numpy as np

from accelerate import Accelerator
from logging import getLogger, Logger
from typing import Tuple

from logadempirical.data.data_loader import process_dataset
from logadempirical.data.vocab import Vocab
from logadempirical.helpers import get_optimizer
from logadempirical.models import get_model, ModelConfig
from logadempirical.trainer import Trainer
from logadempirical.data.log import Log
from logadempirical.data.preprocess import preprocess_data, preprocess_slidings


def run_eval(args, accelerator, logger):
    """
    Load and evaluate a pre-trained model for log anomaly detection.

    Parameters
    ----------
    args (argparse.Namespace): Command-line arguments and configuration.
    accelerator (Accelerator): Hardware accelerator for training.
    logger (Logger): Logger for recording log messages.

    Returns
    ----------
    None
    """
    if args.grouping == "sliding":
        args.output_dir = f"{args.output_dir}{args.dataset_name}/sliding/W{args.window_size}_S{args.step_size}_C{args.is_chronological}_train{args.train_size}"

    else:
        args.output_dir = f"{args.output_dir}{args.dataset_name}/session/train{args.train_size}"

    storeLog = Log(output_dir=args.output_dir)

    train_path, test_path = process_dataset(logger, data_dir=args.data_dir, output_dir=args.output_dir,
                                            log_file=args.log_file,
                                            dataset_name=args.dataset_name, grouping=args.grouping,
                                            window_size=args.window_size, step_size=args.step_size,
                                            train_size=args.train_size, is_chronological=args.is_chronological,
                                            session_type=args.session_level,
                                            storeLog=storeLog)

    os.makedirs(f"{args.output_dir}/vocabs", exist_ok=True)
    vocab_path = f"{args.output_dir}/vocabs/{args.model_name}.pkl"

    log_vocab = build_vocab(vocab_path,
                            args.data_dir,
                            train_path,
                            args.embeddings,
                            embedding_dim=args.embedding_dim,
                            logger=logger)
    model = build_model(args, vocab_size=len(log_vocab))
    eval(args,
         train_path,
         test_path,
         log_vocab,
         model,
         storeLog,
         logger=logger,
         accelerator=accelerator)


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


def build_model(args: argparse.Namespace, vocab_size: int) -> torch.nn.Module:
    """
    Build model
    Parameters
    ----------
    args: argparse.Namespace: Arguments
    vocab_size: int: Size of vocabulary

    Returns
    -------
    model : torch.nn.Module: Model
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


def eval(args: argparse.Namespace,
         train_path: str,
         test_path: str,
         vocab: Vocab,
         model: torch.nn.Module,
         storeLog: Log,
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

    # Build model
    optimizer = get_optimizer(args, model.parameters())
    device = accelerator.device
    model = model.to(device)

    trainer = Trainer(
        model,
        train_dataset=[],
        valid_dataset=[],
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
        f"Loading model from {args.output_dir}/models/{args.model_name}.pt")
    logger.info(f"{model}\n")

    trainer.load_model(f"{args.output_dir}/models/{args.model_name}.pt")

    #  preprocess test data
    test_data, num_sessions = preprocess_data(
        path=test_path,
        args=args,
        is_train=False,
        storeLog=storeLog,
        logger=logger)

    test_dataset, eventIds = preprocess_slidings(
        test_data=test_data,
        vocab=vocab,
        args=args,
        is_train=False,
        storeLog=storeLog,
        logger=logger,
    )

    session_labels = test_dataset.session_labels

    storeLog.lengths
    storeLog.get_train_sliding_window(length=True)
    storeLog.get_valid_sliding_window(length=True)
    storeLog.get_test_sliding_window(length=True)
    # print(
    #     f"eventIds: {len(eventIds)} - session_labels: {len(session_labels)}\n")

    logger.info(
        f"Start predicting {args.model_name} model on {device} device with top-{args.topk} recommendation")
    acc, f1, pre, rec = trainer.predict_unsupervised(test_dataset,
                                                     session_labels,
                                                     eventIds=eventIds,
                                                     topk=args.topk,
                                                     device=device,
                                                     is_valid=False,
                                                     num_sessions=num_sessions,
                                                     storeLog=storeLog
                                                     )

    logger.info(
        f"Test Result:: Acc: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return acc, f1, pre, rec
