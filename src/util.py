"""
Some utilities that are useful
"""


class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    https://stackoverflow.com/a/23689767
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
