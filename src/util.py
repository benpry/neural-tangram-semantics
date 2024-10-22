"""
Some utilities that are useful
"""

import os
from pyprojroot import here
import pickle


class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    https://stackoverflow.com/a/23689767
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self):
        return DotDict(super().copy())


def get_classifiers():
    """
    Get the classifiers available in the sklearn library
    """
    classifiers = []
    for i in range(10):
        filepath = here(f"data/trained-classifiers/mlp_{i}.pkl")
        if os.path.exists(filepath):
            classifiers.append(pickle.load(open(filepath, "rb")))
        else:
            raise FileNotFoundError(f"File {filepath} not found")
