from abc import ABCMeta, abstractmethod, abstractproperty
from sklearn.base import RegressorMixin
import numpy as np


class RegressorMixinND(RegressorMixin):
    def score(self, X, y, sample_weight=None):
        # TODO: multi-output.
        return super().score(X, y.reshape(-1), sample_weight)


class BaseData(metaclass=ABCMeta):
    @abstractmethod
    def check(self):
        """Check validity of the data provided."""


class BaseInputData(BaseData, metaclass=ABCMeta):
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


class StandardInputData(BaseInputData):
    @property
    def n_samples(self):
        pass

    @property
    def n_features(self):
        pass


class PairwiseInputData(BaseInputData):
    def __getitem__(self, indices):
        X_ = X[np.ix_(indices, indices)]
        y_ = y and y[indices]
        return self.__class__(X_, y_)


class InputDataND(BaseInputData):
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
    def n_dimensions(self):
        return len(self.X)
