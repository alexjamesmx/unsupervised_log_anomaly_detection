from pprint import pprint

import os
import pickle


class Log(object):

    def __init__(self,
                 output_dir: str = 'path'):
        self.output_dir = output_dir

        self.logs = None
        self.lengths = {
            "train": None,
            "valid": None,
            "test": None
        }
        self.training_sliding_window = {
            "sequentials": None,
            "quantitivese": None,
            "semantics": None,
            "labels": None,
            "idxs": None,
            "lengths": {
                "sequentials": None,
                "quantitivese": None,
                "semantics": None,
                "labels": None,
                "idxs": None
            }
        }

        self.valid_sliding_window = {
            "sequentials": None,
            "quantitivese": None,
            "semantics": None,
            "labels": None,
            "sequence_idxs": None,
            "session_labels": None,
            "lengths": {
                "sequentials": None,
                "quantitivese": None,
                "semantics": None,
                "labels": None,
                "sequence_idxs": None,
                "session_labels": None
            }
        }
        self.test_sliding_window = {
            "sequentials": None,
            "quantitivese": None,
            "semantics": None,
            "labels": None,
            "sequence_idxs": None,
            "session_labels": None,
            "eventIds": None,
            "lengths": {
                "sequentials": None,
                "quantitivese": None,
                "semantics": None,
                "labels": None,
                "sequence_idxs": None,
                "session_labels": None,
                "eventIds": None
            }
        }

        self.fp_sliding_window = {
            "sequentials": None,
            "quantitivese": None,
            "semantics": None,
            "labels": None,
            "sequence_idxs": None,
            "session_labels": None,
            "eventIds": None,
            "lengths": {
                "sequentials": None,
                "quantitivese": None,
                "semantics": None,
                "labels": None,
                "sequence_idxs": None,
                "session_labels": None,
                "eventIds": None
            }
        }

        self.train_data = []
        self.test_data = []
        self.valid_data = []
        self.original_data = []

        self.false_positives = []
        self.fp_sliding_window = {
            "sequentials": None,
            "quantitatives": None,
            "semantics": None,
            "labels": None,
            "sequence_idxs": None,
            "session_labels": None,
            "lengths": {
                "sequentials":  None,
                "quantitatives":  None,
                "semantics":  None,
                "labels":  None,
                "sequence_idxs":  None,
                "session_labels":  None,
            }
        }

        if os.path.exists(os.path.join(output_dir, "false_positives.pkl")):
            data_path = os.path.join(output_dir, "false_positives.pkl")
            with open(data_path, 'rb') as f:
                self.false_positives = pickle.load(f)
                print("False positives loaded from: ",
                      data_path, "\n")
        else:
            print("False positives not found in: ", output_dir)

    def set_lengths(self, train_length=None, valid_length=None, test_length=None):
        if test_length:
            self.lengths["test"] = test_length
        if valid_length:
            self.lengths["valid"] = valid_length
        if train_length:
            self.lengths["train"] = train_length

    def set_train_sliding_window(self, sequentials=None, quantitatives=None, semantics=None, labels=None, idxs=None):
        if sequentials is None and quantitatives is None and semantics is None:
            raise ValueError('Provide at least one feature type')
        self.training_sliding_window = {
            "sequentials": sequentials,
            "quantitatives": quantitatives,
            "semantics": semantics,
            "labels": labels,
            "idxs": idxs,
            "lengths": {
                "sequentials": len(sequentials) if sequentials is not None else None,
                "quantitatives": len(quantitatives) if quantitatives is not None else None,
                "semantics": len(semantics) if semantics is not None else None,
                "labels": len(labels) if labels is not None else None,
                "idxs": len(idxs) if idxs is not None else None
            }
        }

    def get_train_sliding_window(self, length=False):
        if length:
            print("Train lengths:")
            pprint(self.training_sliding_window["lengths"])
        else:
            print("Train slidings:")
            pprint(self.training_sliding_window)

    def set_valid_sliding_window(self, sequentials=None, quantitatives=None, semantics=None, labels=None,
                                 sequence_idxs=None, session_labels=None):
        if sequentials is None and quantitatives is None and semantics is None:
            raise ValueError('Provide at least one feature type')
        self.valid_sliding_window = {
            "sequentials": sequentials,
            "quantitatives": quantitatives,
            "semantics": semantics,
            "labels": labels,
            "sequence_idxs": sequence_idxs,
            "session_labels": session_labels,
            "lengths": {
                "sequentials": len(sequentials) if sequentials is not None else None,
                "quantitatives": len(quantitatives) if quantitatives is not None else None,
                "semantics": len(semantics) if semantics is not None else None,
                "labels": len(labels) if labels is not None else None,
                "sequence_idxs": len(sequence_idxs) if sequence_idxs is not None else None,
                "session_labels": len(session_labels) if session_labels is not None else None,
            }
        }

    def get_valid_sliding_window(self, length=False, session_labels=False, sequence_idxs=False, labels=False, semantics=False, quantitatives=False, sequentials=False):
        if length:
            print("Valid lengths:")
            return pprint(self.valid_sliding_window["lengths"])
        if session_labels:
            return self.valid_sliding_window["session_labels"]
        if sequence_idxs:
            return self.valid_sliding_window["sequence_idxs"]
        if labels:
            return self.valid_sliding_window["labels"]
        if semantics:
            return self.valid_sliding_window["semantics"]
        if quantitatives:
            return self.valid_sliding_window["quantitatives"]
        else:
            print("Train slidings:")
            return pprint(self.valid_sliding_window)

    def set_test_sliding_window(self, sequentials=None, quantitatives=None, semantics=None, labels=None,
                                sequence_idxs=None, session_labels=None, eventIds=None):
        if sequentials is None and quantitatives is None and semantics is None:
            raise ValueError('Provide at least one feature type')
        self.test_sliding_window = {
            "sequentials": sequentials,
            "quantitatives": quantitatives,
            "semantics": semantics,
            "labels": labels,
            "sequence_idxs": sequence_idxs,
            "lengths": {
                "sequentials": len(sequentials) if sequentials is not None else None,
                "quantitatives": len(quantitatives) if quantitatives is not None else None,
                "semantics": len(semantics) if semantics is not None else None,
                "labels": len(labels) if labels is not None else None,
                "sequence_idxs": len(sequence_idxs) if sequence_idxs is not None else None,

            }
        }

    def get_test_sliding_window(self, length=False):
        if length:
            print("Test lengths:")
            pprint(self.test_sliding_window["lengths"])
        else:
            print("Test slidings:")
            pprint(self.test_sliding_window)

    def set_train_data(self, logs):
        self.train_data = logs

    def set_test_data(self, logs):
        self.test_data = logs

    def set_valid_data(self, logs):
        self.valid_data = logs

    def get_train_data(self, n=None, m=None):
        if n is None:
            n = 0  # Default start value if not provided
        if m is None:
            # Default end value if not provided
            m = len(self.train_data)

        if n < 0:
            n = len(self.train_data) + \
                n  # Handle negative start index

        if m < 0:
            m = len(self.train_data) + \
                m  # Handle negative end index

        if n < 0 or m < 0 or n > len(self.train_data) or m > len(self.train_data) or n > m:
            raise ValueError("Invalid range")

        return self.train_data[n:m]

    def get_test_data(self, n=None, m=None, blockId=None):
        if blockId:
            return [log for log in self.test_data if log[0] == blockId]
        if n is None:
            n = 0  # Default start value if not provided
        if m is None:
            # Default end value if not provided
            m = len(self.test_data)

        if n < 0:
            n = len(self.test_data) + \
                n  # Handle negative start index

        if m < 0:
            m = len(self.test_data) + \
                m  # Handle negative end index

        if n < 0 or m < 0 or n > len(self.test_data) or m > len(self.test_data) or n > m:
            raise ValueError("Invalid range")

        return self.test_data[n:m]

    def get_valid_data(self, n=None, m=None):
        if n is None:
            n = 0  # Default start value if not provided
        if m is None:
            # Default end value if not provided
            m = len(self.valid_data)

        if n < 0:
            n = len(self.valid_data) + \
                n  # Handle negative start index

        if m < 0:
            m = len(self.valid_data) + \
                m  # Handle negative end index

        if n < 0 or m < 0 or n > len(self.valid_data) or m > len(self.valid_data) or n > m:
            raise ValueError("Invalid range")

        return self.valid_data[n:m]

    def set_original_data(self, logs):
        self.original_data = logs

    def get_original_data(self, blockId=None):
        if blockId:
            return [log for log in self.original_data if log["SessionId"] == blockId]
        else:
            return self.original_data

    def set_false_positives(self, logs):
        existing_ids = set(item[0] for item in self.false_positives)

        # Add only the new logs that have unique identifiers
        new_false_positives = [
            log for log in logs if log[0] not in existing_ids]

        # Update the false positives list
        self.false_positives.extend(new_false_positives)

        with open(os.path.join(self.output_dir, "false_positives.pkl"), mode="wb") as f:
            pickle.dump(self.false_positives, f)

    def set_fp_slidings(self, sequentials=None, quantitatives=None, semantics=None, labels=None,
                        sequence_idxs=None, session_labels=None):
        if sequentials is None and quantitatives is None and semantics is None:
            raise ValueError('Provide at least one feature type')
        self.fp_sliding_window = {
            "sequentials": sequentials,
            "quantitatives": quantitatives,
            "semantics": semantics,
            "labels": labels,
            "sequence_idxs": sequence_idxs,
            "session_labels": session_labels,
            "lengths": {
                "sequentials": len(sequentials) if sequentials is not None else None,
                "quantitatives": len(quantitatives) if quantitatives is not None else None,
                "semantics": len(semantics) if semantics is not None else None,
                "labels": len(labels) if labels is not None else None,
                "sequence_idxs": len(sequence_idxs) if sequence_idxs is not None else None,
                "session_labels": len(session_labels) if session_labels is not None else None,
            }
        }

    def get_fp_slidings(self, length=False):
        if length:
            print("False positive lengths:")
            pprint(self.fp_sliding_window["lengths"])
        else:
            print("False positive slidings:")
            pprint(self.fp_sliding_window)
