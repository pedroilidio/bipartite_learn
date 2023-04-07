"""
The :mod:`bipartite_learn.utils` module includes various utilities.
"""
# TODO: reallocate as much as we can.

from __future__ import annotations
import inspect
import pkgutil
import numpy as np
from importlib import import_module
from operator import itemgetter
from pathlib import Path
from typing import Any, Sequence
from sklearn.utils import check_array
from sklearn.utils.validation import check_symmetric
from sklearn.utils.validation import _check_sample_weight
from .discovery import all_estimators

__all__ = [
    "all_estimators",
    "check_multipartite_params",
    "check_simmilarity_matrix",
]


def _check_multipartite_sample_weight(sample_weight, X, **kwargs):
    if not isinstance(sample_weight, np.ndarray):  # Number or None
        row_weight = col_weight = sample_weight
    else:
        n_rows = X[0].shape[0]
        row_weight = sample_weight[:n_rows]
        col_weight = sample_weight[n_rows:]

    row_weight = _check_sample_weight(row_weight, X[0], **kwargs)
    col_weight = _check_sample_weight(col_weight, X[1], **kwargs)

    return np.hstack([row_weight, col_weight])


def _X_is_multipartite(X):
    # TODO: find a better way of deciding.
    return isinstance(X, (tuple, list))


def check_similarity_matrix(
    X,
    *,
    check_symmetry=True,
    symmetry_tol=1e-10,
    symmetry_warning=True,
    symmetry_exception=False,
    **check_array_args,
):
    X = check_array(X, **check_array_args)

    if (X > 1.).any() or (X < 0.).any():
        raise ValueError("Similarity values must be between 0 and 1 "
                         "(inclusive)")

    if check_symmetry:
        return check_symmetric(
            X,
            tol=symmetry_tol,
            raise_warning=symmetry_warning,
            raise_exception=symmetry_exception,
        )
    return X


def check_multipartite_params(*params, k=2):
    new_params = []

    for p in params:
        if isinstance(p, Sequence) and not isinstance(p, str):
            if len(p) != k:
                raise ValueError(
                    f"Parameter {p} was required to have length={k}")
            new_params.append(p)
        else:
            new_params.append([p for _ in range(k)])
    
    return new_params[0] if len(params) == 1 else new_params


def _partiteness_name(k: int) -> str:
    if k < 1:
        raise ValueError("Invalid partiteness.")
    elif k == 1:
        return "monopartite"
    elif k == 2:
        return "bipartite"
    elif k == 3:
        return "tripartite"
    else:
        return f"{k}-partite"


def check_partiteness(
    X=None,
    y=None,
    *,
    partiteness: None | int = None,
    estimator: None | str | Any = None,
):
    if estimator is not None and not isinstance(estimator, str):
        if hasattr(estimator, '_partiteness'):
            partiteness = estimator._partiteness
            if partiteness is None:  # Estimator accepts any partiteness
                return
        elif partiteness is None:
            raise ValueError("If no partiteness provided, estimator must have "
                             "a 'partiteness' attribute.")
    elif partiteness is None:
        raise ValueError(
            "Either partiteness or an estimator instance must be provided."
        )

    kpartite = _partiteness_name(partiteness)

    if estimator is None:
        but_text = f"but {kpartite} input was expected."
    elif isinstance(estimator, str):
        but_text = f"but {estimator} expects {kpartite} input."
    else:
        but_text = f"but {type(estimator).__name__} expects {kpartite} input."

    if X is None and y is None:
        raise ValueError("Either X or y must be provided.")

    if X is not None:
        len_X = len(X)
        if len_X != partiteness:
            raise ValueError(
                f"X is {_partiteness_name(len_X)} ({len(X)=}) " + but_text
            )
    if y is not None:
        if y.ndim != partiteness:
            raise ValueError(
                f"y is {_partiteness_name(y.ndim)} ({y.ndim=}) " + but_text
            )