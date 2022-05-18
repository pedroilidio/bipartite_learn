import numpy as np
from imblearn.base import BaseSampler


class Melter2D(BaseSampler):
    """Convert a 2D GMO database to a simpler GSO database.

    Convert a 2D interaction problem, where there are two feature matrices in X,
    (one for each axis) and an interaction matrix y to a simpler usual format
    where each sample is a combination of samples from X[0] and X[1].
    """
    def fit_resample(self, X, y, W=1, seed=None):
        # FIXME: we are bypassing input checking.
        return self._fit_resample(X, y)

    def _fit_resample(self, X, y):
        X1, X2 = X
        melted_X = np.hstack([
            np.repeat(X1, X2.shape[0], axis=0),
            np.tile(X2, (X1.shape[0], 1))
        ])
        melted_y = y.reshape(-1)
        return melted_X, melted_y


class MelterND(BaseSampler):
    """Convert a n-dimensional GMO database to a simpler GSO database."""
    def fit_resample(self, X, y, W=1, seed=None):
        # FIXME: we are bypassing input checking.
        return self._fit_resample(X, y)

    def _fit_resample(self, X, y=None):
        melted_X = row_cartesian_product(X)
        if y is None:
            return melted_X
        melted_y = y.reshape(-1)
        return melted_X, melted_y


def row_cartesian_product(X):
    """Row cartesian product of 2D arrays in X.

    Pick one row from each of the 2D arrays in X, in their presented order, and
    concatenate them. Repeat. Return a 2D array where its rows are all the
    possible combinations of rows in X.

    Parameters
    ----------
    X : list-like of 2D np.ndarrays

    Returns
    -------
    result : 2D np.ndarray
        Cartesian product of X's 2d arrays, row-wise.
    """
    lengths = [Xi.shape[0] for Xi in X]
    # Xi will repeat once for every combination of samples ahead (Xj if j>i)
    # and tile once for every combination of samples before (Xj if j<i).
    n_repeat_tile = [
        (np.prod(lengths[(i+1):]), np.prod(lengths[:i]))
        for i in range(len(X))
    ]

    result = np.hstack([
        np.tile(Xi.repeat(n_repeat, axis=0), (n_tile, 1))
        for Xi, (n_repeat, n_tile) in zip(X, n_repeat_tile)
    ])

    return result
