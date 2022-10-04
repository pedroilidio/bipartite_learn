"""
The :mod:`hypertrees.utils` module includes various utilities.
"""
# TODO: reallocate as much as we can.

from __future__ import annotations
import inspect
import pkgutil
import numpy as np
from importlib import import_module
from operator import itemgetter
from pathlib import Path
from typing import Callable, Sequence
from sklearn.utils import IS_PYPY

__all__ = [
    "all_estimators",
    "check_multipartite_params",
]


def lazy_knn_weights(
    distances: np.ndarray[float],
    min_distance: float = 0.,
    func: Callable = lambda x: 1/x,
    **func_kwargs,
):
    """Return learned instance if KNN estimator finds one during predict.
    
    Intended to be used as the `weights` parameter of KNN estimators such as
    `sklearn.neighbors.KNeighborsRegressor`. If an instance given to predict
    is identical to an instance of the training set, the KNN estimator will
    not average its neighbors' labels, as usual, but will simply return the
    known instance's labels.

    Parameters
    ----------
    distances : np.ndarray[float]
        The distance matrix to calculate weights on.
    min_distance : float, default=0.
        Minimum distance to consider samples identical.
    func : Callable, optional, default=lambda x: 1/x
        Function to apply on remaining instances, not found in the training set.
    **func_kwargs : dict
        Other keyword arguments will be forwarded to func.

    Returns
    -------
    np.ndarray with same shape as `distances`
        Weight matrix, with known instances only having weight on the columns
        corresponding to its positions (if repeated) on the training set. The
        remaining weights of known instances are set to zero.
    """
    is_known = (distances <= min_distance).any(axis=1)
    weights = np.empty_like(distances)

    # Only the known instance's column has non-zero weight.
    weights[is_known] = (distances[is_known] <= min_distance)

    # Remaing weights are calculated by `func`.
    weights[~is_known] = func(distances[~is_known], **func_kwargs)

    return weights


def lazy_knn_weights_min_one(*args, **kwargs):
    """Return learned instance if KNN estimator finds one during predict.

    Convenience partial function to call `lazy_knn_weights()` with
    `min_distace=1`, since it's so common when dealing with inverses of
    similarities in the 0 to 1 interval.

    See also
    --------
    lazy_knn_weights
    """
    return lazy_knn_weights(*args, min_distance=1., **kwargs)


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


def all_estimators(type_filter=None):
    """Get a list of all estimators from hypertrees.

    This function is adapted from the original sklearn.utils.all_estimators to
    crawl the hypertrees package.

    This function crawls the module and gets all classes that inherit
    from sklearn.BaseEstimator. Classes that are defined in test-modules
    are not included.

    Parameters
    ----------
    type_filter : {"classifier", "regressor", "cluster", "transformer"} \
            or list of such str, default=None
        Which kind of estimators should be returned. If None, no filter is
        applied and all estimators are returned.  Possible values are
        'classifier', 'regressor', 'cluster' and 'transformer' to get
        estimators only of these specific types, or a list of these to
        get the estimators that fit at least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.
    """
    # lazy import to avoid circular imports from sklearn.base
    from sklearn.utils._testing import ignore_warnings
    from sklearn.base import (
        BaseEstimator,
        ClassifierMixin,
        RegressorMixin,
        TransformerMixin,
        ClusterMixin,
    )

    def is_abstract(c):
        if not (hasattr(c, "__abstractmethods__")):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    modules_to_ignore = {
        "tests",
        "externals",
        "setup",
        "conftest",
        "enable_hist_gradient_boosting",
    }
    root = str(Path(__file__).parent.parent)  # hypertrees package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=[root], prefix="hypertrees."
        ):
            mod_parts = modname.split(".")
            if any(part in modules_to_ignore for part in mod_parts) or "._" in modname:
                continue
            module = import_module(modname)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, est_cls) for name, est_cls in classes if not name.startswith("_")
            ]

            # TODO: Remove when FeatureHasher is implemented in PYPY
            # Skips FeatureHasher for PYPY
            if IS_PYPY and "feature_extraction" in modname:
                classes = [
                    (name, est_cls)
                    for name, est_cls in classes
                    if name == "FeatureHasher"
                ]

            all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [
        c
        for c in all_classes
        if (issubclass(c[1], BaseEstimator) and c[0] != "BaseEstimator")
    ]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {
            "classifier": ClassifierMixin,
            "regressor": RegressorMixin,
            "transformer": TransformerMixin,
            "cluster": ClusterMixin,
        }
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend(
                    [est for est in estimators if issubclass(est[1], mixin)]
                )
        estimators = filtered_estimators
        if type_filter:
            raise ValueError(
                "Parameter type_filter must be 'classifier', "
                "'regressor', 'transformer', 'cluster' or "
                "None, got"
                " %s."
                % repr(type_filter)
            )

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))
