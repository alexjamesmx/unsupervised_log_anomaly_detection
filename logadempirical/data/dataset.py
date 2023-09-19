# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset


class LogDataset(Dataset):
    def __init__(self, sequentials=None, quantitatives=None, semantics=None, labels=None, idxs=None, session_labels=None):
        if sequentials is None and quantitatives is None and semantics is None:
            raise ValueError('Provide at least one feature type')
        self.sequentials = sequentials
        self.quantitatives = quantitatives
        self.semantics = semantics
        self.labels = labels
        self.idxs = idxs
        self.session_labels = session_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {'label': self.labels[idx], 'idx': self.idxs[idx]}
        if self.sequentials is not None:
            item['sequential'] = torch.from_numpy(
                np.array(self.sequentials[idx]))
        if self.quantitatives is not None:
            item['quantitative'] = torch.from_numpy(
                np.array(self.quantitatives[idx], )[:, np.newaxis]).float()
        if self.semantics is not None:
            item['semantic'] = torch.from_numpy(
                np.array(self.semantics[idx])).float()

        return item

    def get_sequential(self):
        return self.sequentials

    def get_quantitative(self):
        return self.quantitatives

    def get_semantic(self):
        return self.semantics

    def get_label(self):
        return self.labels

    def get_idx(self):
        return self.idxs

    def get_session_labels(self):
        return self.session_labels


def data_collate(batch, feature_name='semantic', padding_side="right"):
    max_length = max([len(b[feature_name]) for b in batch])
    dimension = {k: batch[0][k][0].shape[0] for k in batch[0].keys(
    ) if k != 'label' and batch[0][k] is not None}
    if padding_side == "right":
        padded_batch = []
        for b in batch:
            sample = {}
            for k, v in b.items():
                if k == 'label':
                    sample[k] = v
                elif v is None:
                    sample[k] = None
                else:
                    sample[k] = torch.from_numpy(
                        np.array(v + [np.zeros(dimension[k], )] * (max_length - len(v))))
            padded_batch.append(sample)
    elif padding_side == "left":
        padded_batch = []
        for b in batch:
            sample = {}
            for k, v in b.items():
                if k == 'label':
                    sample[k] = v
                elif v is None:
                    sample[k] = None
                else:
                    sample[k] = torch.from_numpy(
                        np.array([np.zeros(dimension[k], )] * (max_length - len(v)) + v))
            padded_batch.append(sample)
    else:
        raise ValueError("padding_side should be either 'right' or 'left'")

    # convert to tensor
    padded_batch = {
        k: torch.stack([sample[k] for sample in padded_batch])
        for k in padded_batch[0].keys()
    }
    return padded_batch
