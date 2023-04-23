"""Experimental interfaces for datasets.
"""
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np


class BaseData(metaclass=ABCMeta):
    @abstractmethod
    def check(self):
        """Check validity of the data provided."""


class BaseDataset(BaseData, metaclass=ABCMeta):
    def __init__(self, X, y=None, check=False):
        self.X = X
        self.y = y
        self.Xy = X, y
        if check:
            self.check()

    @abstractmethod
    def __getitem__(*indices):
        ...
    
    def check(self, estimator=None):
        """Check validity of the data provided."""
        return self

    @abstractproperty
    def n_samples(self):
        """The number of samples in the dataset."""

    @abstractproperty
    def n_features(self):
        """The number of features in the dataset."""


class StandardDataset(BaseDataset):
    def __getitem__(self, indices):
        return self.__class__(
            X=self.X[indices],
            y=self.y and self.y[indices],
        )

    @property
    def n_samples(self):
        return self.X.shape[0]

    @property
    def n_features(self):
        return self.X.shape[1]

    @property
    def n_outputs(self):
        return self.y.shape[1]


class PairwiseDataset(BaseDataset):
    def __getitem__(self, indices):
        return self.__class__(
            X=self.X[np.ix_(indices, indices)],
            y=self.y and self.y[indices],
        )


class BipartiteDataset(BaseDataset):
    def __getitem__(self, indices):
        X_ = [Xi[np.r_[idx]] for Xi, idx in zip(self.X, indices)]
        if self.y is None:
            y_ = None
        else:
            y_ = self.y[np.ix_(*(np.r_[idx] for idx in indices))]
        return self.__class__(X_, y_)

    @property
    def n_samples(self):
        # TODO: Use sklearn's _n_samples()
        return tuple(Xi.shape[0] for Xi in self.X)

    @property
    def n_features(self):
        return tuple(Xi.shape[1] for Xi in self.X)

    @property
    def n_parts(self):
        return len(self.X)