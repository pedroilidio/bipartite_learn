import numpy as np
from sklearn.utils.validation import check_random_state
from sklearn.utils._param_validation import validate_params
from bipartite_learn.base import (
    BaseBipartiteEstimator,
    BaseMultipartiteSampler,
)
from bipartite_learn.utils import _X_is_multipartite

__all__ = [
    "BipartiteMelter",
    "MultiPartiteMelter",
    "melt_multipartite_dataset",
]


class BipartiteMelter(BaseMultipartiteSampler, BaseBipartiteEstimator):
    """Convert a bipartite GMO database to a simpler GSO database.

    Convert a bipartite interaction problem, where there are two feature
    matrices in X (one for each axis) and an interaction matrix y to a simpler
    usual format where each sample is a combination of samples from X[0] and
    X[1].

    Slightly faster than MultipartiteMelter
    """

    def _fit_resample(self, X, y):
        X1, X2 = X
        melted_X = np.hstack([
            np.repeat(X1, X2.shape[0], axis=0),
            np.tile(X2, (X1.shape[0], 1))
        ])
        melted_y = y.reshape(-1)
        return melted_X, melted_y


class MultipartiteMelter(BaseMultipartiteSampler):
    """Convert a n-dimensional GMO database to a simpler GSO database."""

    def __init__(self, subsample_negatives=False, random_state=None):
        self.subsample_negatives = subsample_negatives
        self.random_state = random_state

    def _fit_resample(self, X, y=None):
        return melt_multipartite_dataset(
            X,
            y,
            subsample_negatives=self.subsample_negatives,
            # Delegating check_random_state since samplers are stateless
            random_state=self.random_state,
        )


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


@validate_params({
    'X': ['array-like'],
    'y': ['array-like', None],
    'subsample_negatives': ['boolean'],
    'random_state': ['random_state'],
})
def melt_multipartite_dataset(
    X, y=None, subsample_negatives=False, random_state=None,
):
    """Melt bipartite input.
    
    If X is a list of Xi feature matrices, one for each bipartite group,
    convert it to traditional data format by generating concatenations of
    rows from X[0] with rows from X[1].
    """
    if not _X_is_multipartite(X):
        raise ValueError(
            "Tried to melt monopartite dataset. X must be a list/tuple of "
            "array-likes."
        )
    X = row_cartesian_product(X) 
    if y is None:
        return X, y

    y = y.reshape(-1)

    if subsample_negatives:
        if set(np.unique(y)) != {0, 1}:
            raise ValueError(
                "y must have only 0 or 1 elements if subsample_negatives=True."
            )

        random_state = check_random_state(random_state)
        pos_negatives = np.nonzero(1 - y)[0]
        # difference in number of positives vs number of negatives
        # i. e. number of negatives minus number of positives
        delta = 2 * pos_negatives.size - y.size

        if delta < 0:
            raise ValueError(
                "There must be less positive than negative values to subsample"
                "negatives."
            )

        # Subsample negatives to have the same size as positives
        negatives_to_del = random_state.choice(
            pos_negatives,
            replace=False,
            size=delta,
        )
        X = np.delete(X, negatives_to_del, axis=0)
        y = np.delete(y, negatives_to_del, axis=0)
    
    return X, y

