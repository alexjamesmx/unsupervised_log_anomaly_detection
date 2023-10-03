from collections import Counter
import pickle
import json

import numpy as np
from numpy import dot
from numpy.linalg import norm
import math


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


class Vocab(object):
    def __init__(self, logs, emb_file="embeddings.json", embedding_dim=100):
        self.emedding_dim = embedding_dim
        self.stoi = {}
        self.itos = ['padding']
        self.pad_token = "padding"
        # NOTE as logs are a list of lists, we need to flatten it first and remove duplicates
        for line in logs:
            self.itos.extend(line)

        self.itos = ['padding'] + list(set(self.itos))

        # self.pad_index = len(self.itos)
        # self.itos.append("padding")
        self.unk_index = len(self.itos)
        # NOTE add indices where e is the event and i is the index
        self.stoi = {e: i for i, e in enumerate(self.itos)}
        self.semantic_vectors = read_json(emb_file)
        self.semantic_vectors = {k: v if type(v) is list else [0] * embedding_dim
                                 for k, v in self.semantic_vectors.items()}
        # NOTE add token at the end of the vocab
        self.semantic_vectors[self.pad_token] = [-1] * embedding_dim
        self.mapping = {}

        self.save_path = ""

    def __len__(self):
        return len(self.itos)

    def get_event(self, real_event, use_similar=False):
        event = self.stoi.get(real_event, self.unk_index)
        if not use_similar or event != self.unk_index:
            return event
        if self.mapping.get(real_event) is not None:
            return self.mapping[real_event]

        for train_event in self.itos[:-1]:
            sim = dot(self.semantic_vectors[real_event], self.semantic_vectors[train_event]) / (norm(
                self.semantic_vectors[real_event]) * norm(self.semantic_vectors[train_event]))
            if sim > 0.90:
                self.mapping[real_event] = self.stoi.get(train_event)
                return self.stoi.get(train_event)
        self.mapping[real_event] = self.unk_index
        return self.mapping[real_event]

    def get_embedding(self, event):
        return self.semantic_vectors[event]

    def update_vocab(self, new_event):
        if new_event not in self.stoi:
            self.itos.append(new_event)
            self.stoi[new_event] = len(self.stoi) + 1
            # TODO udpate semantic vectors
            # self.semantic_vectors[new_event] = [0] * self.emedding_dim
            # self.mapping[new_event] = self.stoi[new_event]
        updated_path = self.save_path.replace(".pkl", "_updated.pkl")
        # t = self.save_path.rsplit('/', 1)[-1]
        # p = t.rsplit('.', 1)[0]
        # save_path = self.save_path.replace(p, p + "_updated")
        print(f"Save updated vocab in {updated_path}")
        with open(updated_path, 'wb') as f:
            pickle.dump(self, f)

    def save_vocab(self, file_path):
        self.save_path = file_path
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_vocab(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
