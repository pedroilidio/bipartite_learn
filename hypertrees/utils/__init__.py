"""
The :mod:`hypertrees.utils` module includes various utilities.
"""
# TODO: reallocate as much as we can.

from __future__ import annotations
import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path
from typing import Any, Sequence
from sklearn.utils import check_array, IS_PYPY
from sklearn.utils.validation import check_symmetric

__all__ = [
    "all_estimators",
    "check_multipartite_params",
    "check_simmilarity_matrix",
]


def _X_is_multipartite(X):
    # TODO: find a better way of deciding.
    return isinstance(X, (tuple, list))


def check_similarity_matrix(
    X,
    symmetry_tol=1e-10,
    symmetry_warning=True,
    symmetry_exception=False,
    **check_array_args,
):
    X = check_array(X, **check_array_args)

    if (X > 1.).any() or (X < 0.).any():
        raise ValueError("Similarity values must be between 0 and 1 "
                         "(inclusive)")

    return check_symmetric(
        X,
        tol=symmetry_tol,
        raise_warning=symmetry_warning,
        raise_exception=symmetry_exception,
    )


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
