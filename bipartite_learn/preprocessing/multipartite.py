# TODO: docs
# TODO: KNN sampler to be used by NRLMF
# TODO: Beta correction kernel aggregator
# TODO: MultipartiteSamplerWrapper
from __future__ import annotations
from numbers import Number
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from ..base import BaseMultipartiteSampler


class DTHybridSampler(BaseMultipartiteSampler):
    def __init__(self, lamb=.5, alpha=.5):
        self.lamb = lamb
        self.alpha = alpha

    def _fit_resample(self, X, y):
        S_row_net = y @ y.T  # Rows network similarity
        k_row = np.diag(S_row_net).copy()  # Row nodes degree
        # If the row makes no interactions, set the similarity to 0
        # (bellow we will divide the similarity by k_row)
        # TODO: raise warning.
        k_row[k_row == 0] = np.inf

        # k_row_total[i, j] = k_row[i] + k_row[j]
        k_row_total = k_row[:, np.newaxis] * k_row

        # k_row_weighted[i, j] = lambda**k_row[i] + k_row[j]**(1-lambda)
        k_row_weighted = (
            k_row[:, np.newaxis]**self.lamb * k_row**(1-self.lamb)
        )

        k_col = y.sum(axis=0)  # Column nodes degree
        # If the column makes no interactions, set the similarity to 0
        k_col[k_col == 0] = np.inf

        # Row similarity based on column chemical similarity
        S_row_from_column = y @ X[1] @ y.T / k_row_total

        S = self.alpha * X[0] + (1-self.alpha) * S_row_from_column
        Gamma = S / k_row_weighted

        W = Gamma * ((y / k_col) @ y.T)

        return X, W @ y
    