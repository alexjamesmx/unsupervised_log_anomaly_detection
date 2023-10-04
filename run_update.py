import os
import pickle
import torch
import argparse
import numpy as np

from accelerate import Accelerator
from typing import Tuple
from logging import getLogger, Logger

from logadempirical.data.data_loader import process_dataset
from logadempirical.data.vocab import Vocab
from logadempirical.data.feature_extraction import load_features, sliding_window
from logadempirical.data.dataset import LogDataset
from logadempirical.helpers import arg_parser, get_optimizer
from logadempirical.models import get_model, ModelConfig
from logadempirical.trainer import Trainer
from logadempirical.data.log import Log
from logadempirical.data.preprocess import preprocess_data, preprocess_slidings


def run_update(args, accelerator, logger):
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

    update(args,
           train_path,
           test_path,
           log_vocab,
           model,
           storeLog=storeLog,
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
    updated_path = vocab_path.replace(".pkl", "_updated.pkl")
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
        if os.path.exists(updated_path):
            vocab_path = updated_path
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
    model: torch.nn.Module: Model

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


def update(args: argparse.Namespace,
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
    # PREPROCESSING
    # save test data
    preprocess_data(
        path=test_path,
        args=args,
        is_train=False,
        storeLog=storeLog,
        logger=logger)

    false_positive_data = storeLog.get_test_data(
        blockId="blk_7608151113726761274")
    # if_unk_vocab_size_before = len(vocab)
    toList = list(false_positive_data[0])
    toList[2] = 0
    false_positive_data[0] = tuple(toList)
    print(f"false positive data: {false_positive_data[0]}")

    print(f"label of false positive data: {false_positive_data[0][2]}")
    sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels = sliding_window(
        false_positive_data,
        vocab=vocab,
        window_size=args.history_size,
        is_train=True,
        is_update=True,
        semantic=args.semantic,
        quantitative=args.quantitative,
        sequential=args.sequential,
        logger=logger
    )

    # if_unk_vocab_size_after = len(vocab)

    # print(
    #     f"Vocab size before: {if_unk_vocab_size_before}, Vocab size after: {if_unk_vocab_size_after}")
    # model = build_model(args, vocab_size=len(vocab))
    # model.to(device)
    # trainer = Trainer(
    #     model,
    #     train_dataset=[],
    #     valid_dataset=[],
    #     is_train=True,
    #     optimizer=optimizer,
    #     no_epochs=args.max_epoch,
    #     batch_size=args.batch_size,
    #     scheduler_type=args.scheduler,
    #     warmup_rate=args.warmup_rate,
    #     accumulation_step=args.accumulation_step,
    #     logger=logger,
    #     accelerator=accelerator,
    #     num_classes=len(vocab),
    # )

    # trainer.save_model(save_dir=f"{args.output_dir}/models",
    #                    model_name=args.model_name,)
    # if (if_unk_vocab_size_before != if_unk_vocab_size_after):
    #     model = build_model(args, vocab_size=len(vocab))
    #     trainer = Trainer(
    #         model,
    #         train_dataset=[],
    #         valid_dataset=[],
    #         is_train=True,
    #         optimizer=optimizer,
    #         no_epochs=args.max_epoch,
    #         batch_size=args.batch_size,
    #         scheduler_type=args.scheduler,
    #         warmup_rate=args.warmup_rate,
    #         accumulation_step=args.accumulation_step,
    #         logger=logger,
    #         accelerator=accelerator,
    #         num_classes=len(vocab),
    #     )
    #     print("bien")
    #     trainer.save_model(save_dir=f"{args.output_dir}/models",
    #                        model_name=args.model_name,)
    #     print('mal')

    false_positive_dataset = LogDataset(
        sequentials, quantitatives, semantics, labels, sequence_idxs, session_labels=session_labels)

    storeLog.set_fp_slidings(sequentials=sequentials,
                             quantitatives=quantitatives,
                             semantics=semantics,
                             labels=labels,
                             sequence_idxs=sequence_idxs,
                             session_labels=session_labels)
    storeLog.get_fp_slidings()
    # print(vocab.stoi)
    # TRAIN
    device = accelerator.device
    model = model.to(device)
    logger.info(
        f"Loading model from {args.output_dir}/models/{args.model_name}.pt{model}\n")
    logger.info(
        f"Start update training {args.model_name} model on {device} device\n")
    optimizer = get_optimizer(args, model.parameters())

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

    trainer.load_model(
        f"{args.output_dir}/models/{args.model_name}.pt")

    train_loss = 0
    # train_loss, args.topk = trainer.train_on_false_positive(false_positive_dataset=false_positive_dataset,
    #                                                         device=device,
    #                                                         save_dir=f"{args.output_dir}/models",
    #                                                         model_name=args.model_name,
    #                                                         topk=args.topk)
    # logger.info(f"Updated Train Loss: {train_loss:.4f}")

    storeLog.set_false_positives(false_positive_data)

    print(storeLog.false_positives)
