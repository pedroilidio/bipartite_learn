import numpy as np
from sklearn.utils.validation import check_random_state
from sklearn.utils._param_validation import validate_params
from bipartite_learn.base import (
    BaseBipartiteEstimator,
    BaseMultipartiteSampler,
)
from bipartite_learn.utils import _X_is_multipartite

__all__ = [
    "row_cartesian_product",
    "melt_multipartite_dataset",
    "BipartiteMelter",
    "MultiPartiteMelter",
]


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

    return np.hstack([
        np.tile(Xi.repeat(n_repeat, axis=0), (n_tile, 1))
        for Xi, (n_repeat, n_tile) in zip(X, n_repeat_tile)
    ])


@validate_params({'X': [list, tuple], 'y': ['array-like', None]})
def melt_multipartite_dataset(X, y=None):
    """Melt bipartite input.
    
    If X is a list of Xi feature matrices, one for each bipartite group,
    convert it to traditional data format by generating concatenations of
    rows from X[0] with rows from X[1].
    """
    if not _X_is_multipartite(X):
        raise ValueError("Tried to melt monopartite dataset.")

    X = row_cartesian_product(X) 

    if y is None:
        return X, y

    return X, y.reshape(-1, 1)


class BipartiteMelter(BaseMultipartiteSampler, BaseBipartiteEstimator):
    """Convert a bipartite dataset to a simpler global-single output format.

    Convert a bipartite interaction problem, where there are two feature
    matrices in X (one for each axis) and an interaction matrix y to a simpler
    usual format where each sample is a combination of samples from X[0] and
    X[1].

    Slightly faster than MultipartiteMelter.
    """

    def _fit_resample(self, X, y):
        X1, X2 = X
        melted_X = np.hstack([
            np.repeat(X1, X2.shape[0], axis=0),
            np.tile(X2, (X1.shape[0], 1))
        ])
        return melted_X, y.reshape(-1, 1)


class MultipartiteMelter(BaseMultipartiteSampler):
    """Convert a multipartite dataset to a simpler global-single output format.
    """

    def _fit_resample(self, X, y):
        return melt_multipartite_dataset(X, y)