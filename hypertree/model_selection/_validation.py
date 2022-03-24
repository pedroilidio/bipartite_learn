"""
The :mod:`sklearn.model_selection._validation` module includes classes and
functions to validate the model.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>
#         Michal Karbownik <michakarbownik@gmail.com>
# ND adapted by Pedro Ilídio.
# License: BSD 3 clause

# New imports:
import sklearn.utils
import copy
import itertools
from sklearn.model_selection._validation import (
    _score,
    _normalize_score_results,
    _warn_about_fit_failures,
    _aggregate_score_dicts,
)
from sklearn.utils.validation import (
    _check_fit_params,
    _make_indexable,
    _num_samples,
)
from sklearn.utils._tags import _safe_tags

import warnings
import numbers
import time
from traceback import format_exc
from contextlib import suppress
from collections import Counter

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, logger

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable, check_random_state, _safe_indexing
from sklearn.utils.fixes import delayed
from sklearn.utils.metaestimators import _safe_split
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._split import check_cv
from sklearn.preprocessing import LabelEncoder


__all__ = [
    "cross_validate_nd",
]


def cross_validate_nd(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    error_score=np.nan,
    diagonal=False,
    train_test_combinations=None,
):
    # TODO: ND adapt.
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Read more in the :ref:`User Guide <multimetric_cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    diagonal : bool, default=False
        Wether to combine each axis splits in a product-manner or zip-manner.

    scoring : str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        See :ref:`multimetric_grid_search` for an example.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`.Fold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    return_train_score : bool, default=False
        Whether to include train scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

        .. versionadded:: 0.19

        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``

    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.

        .. versionadded:: 0.20

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

        .. versionadded:: 0.20

    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.

        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:

            ``test_score``
                The score array for test scores on each cv split.
                Suffix ``_score`` in ``test_score`` changes to a specific
                metric like ``test_r2`` or ``test_auc`` if there are
                multiple scoring metrics in the scoring parameter.
            ``train_score``
                The score array for train scores on each cv split.
                Suffix ``_score`` in ``train_score`` changes to a specific
                metric like ``train_r2`` or ``train_auc`` if there are
                multiple scoring metrics in the scoring parameter.
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``
            ``estimator``
                The estimator objects for each cv split.
                This is available only if ``return_estimator`` parameter
                is set to ``True``.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.metrics import make_scorer
    >>> from sklearn.metrics import confusion_matrix
    >>> from sklearn.svm import LinearSVC
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()

    Single metric evaluation using ``cross_validate``

    >>> cv_results = cross_validate(lasso, X, y, cv=3)
    >>> sorted(cv_results.keys())
    ['fit_time', 'score_time', 'test_score']
    >>> cv_results['test_score']
    array([0.3315057 , 0.08022103, 0.03531816])

    Multiple metric evaluation using ``cross_validate``
    (please refer the ``scoring`` parameter doc for more information)

    >>> scores = cross_validate(lasso, X, y, cv=3,
    ...                         scoring=('r2', 'neg_mean_squared_error'),
    ...                         return_train_score=True)
    >>> print(scores['test_neg_mean_squared_error'])
    [-3635.5... -3573.3... -6114.7...]
    >>> print(scores['train_r2'])
    [0.28009951 0.3908844  0.22784907]

    See Also
    --------
    cross_val_score : Run cross-validation for single metric evaluation.

    cross_val_predict : Get predictions from each split of cross-validation for
        diagnostic purposes.

    sklearn.metrics.make_scorer : Make a scorer from a performance metric or
        loss function.

    """

    # *X, y, groups = indexable(*X, y, groups)
    # Call _make_indexable on each one instead of indexable. The only difference
    # being we avoid calling sklear.utils.check_consistent_length this way.
    X = [_make_indexable(a) for a in X]
    y = _make_indexable(y)

    ndim = y.ndim

    if groups is None:
        groups = [None] * ndim
    groups = [_make_indexable(a) for a in groups]

    # Check dimension consistency
    if not (ndim == len(X) == len(groups)):
        # FIXME: multi-output: y would have an extra dimension.
        raise ValueError("Incompatible dimensions. One must ensure "
                         "y.ndim == len(X) == len(groups)")

    if type(cv) not in (tuple, list):
        cv = [copy.deepcopy(cv) for _ in range(ndim)]

    cv = [check_cv(cv_i, y, classifier=is_classifier(estimator))
          for cv_i in cv]

    if diagonal:
        # Ensure all cross validators report the same number of splits.
        n_splits = (cv[ax].get_n_splits(X[ax], y.moveaxis(ax, 0), groups[ax])
                    for ax in range(ndim))
        g = itertools.groupby(n_splits)
        next(g, None)

        # if not all elements in n_splits are equal:
        if next(g, False):
            raise ValueError("Cross-validators must generate the same number"
                             " of splits if diagonal=True")

    train_test_combinations, train_test_names = _check_train_test_combinations(
        train_test_combinations,
        len(X),
        return_train_score,
    )

    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(estimator, scoring)
    else:
        scorers = _check_multimetric_scoring(estimator, scoring)

    combine_func = zip if diagonal else itertools.product

    splits_iter = combine_func(*(
        # FIXME: is np.moveaxis well suited here?
        cv[ax].split(X[ax], np.moveaxis(y, ax, 0), groups[ax])
        for ax in range(ndim)
    ))

    # next(splits_iter) == train_test ==
    #   ((train_ax0, test_ax0), (train_ax1, test_ax1), ...)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(_fit_and_score_nd)(
            clone(estimator),
            X,
            y,
            scorers,
            train_test,
            verbose,
            None,
            fit_params,
            return_train_score=return_train_score,
            return_times=True,
            return_estimator=return_estimator,
            error_score=error_score,
            train_test_combinations=train_test_combinations,
            train_test_names=train_test_names,
        )
        for train_test in splits_iter
    )

    _warn_about_fit_failures(results, error_score)

    # For callabe scoring, the return type is only know after calling. If the
    # return type is a dictionary, the error scores can now be inserted with
    # the correct key.
    # 
    # if callable(scoring):  # TODO
    #     _insert_error_scores(results, error_score)

    results = _aggregate_score_dicts(results)

    ret = {}
    ret["fit_time"] = results["fit_time"]
    ret["score_time"] = results["score_time"]

    if return_estimator:
        ret["estimator"] = results["estimator"]

    for ttc_name in train_test_names:
        test_scores_dict = _normalize_score_results(
                                results[f"{ttc_name}_scores"])

        for name in test_scores_dict:
            ret[f"{ttc_name}_{name}"] = test_scores_dict[name]

    return ret


def _fit_and_score_nd(
    estimator,
    X,
    y,
    scorer,
    train_test,
    verbose,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
    train_test_combinations=None,
    train_test_names=None,
):

    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like of shape (n_train_samples,)
        Indices of training samples.

    test : array-like of shape (n_test_samples,)
        Indices of test samples.

    verbose : int
        The verbosity level.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : bool, default=False
        Compute and return score on training set.

    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.

    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).

    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).

    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.

    return_times : bool, default=False
        Whether to return the fit/score times.

    return_estimator : bool, default=False
        Whether to return the fitted estimator.

    Returns
    -------
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_error : str or None
            Traceback str if the fit failed, None if the fit succeeded.
    """
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    # fit_params = _check_fit_params(X, fit_params, train)  # FIXME

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    ####################### MODFIED SECTION ##########################

    test_splits = {}

    # NOTE: ttc stands for train-test combinations
    for is_test_tuple, ttc_name in zip(train_test_combinations, train_test_names):
        # is_test_tuple ~= (0, 1, 1, 0)
        test_indices = [ax_train_test[is_test] for is_test, ax_train_test in
                        zip(is_test_tuple, train_test)]
        test_splits[ttc_name] = _safe_split_nd(estimator, X, y, test_indices)


    train_indices = [i[0] for i in train_test]
    X_train, y_train = _safe_split_nd(estimator, X, y, train_indices)

    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {
                    ttc_name+'_scores': {name: error_score for name in scorer}
                    for ttc_name in test_splits.keys()
                }
                # NOTE: train_score is automatically included by
                # _check_train_test_combinations, if requested.
            else:
                test_scores = {
                    ttc_name+'_scores': error_score
                    for ttc_name in test_splits.keys()
                }
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        test_scores = {}
        for ttc_name, (X_test, y_test) in test_splits.items():
            test_scores[ttc_name+'_scores'] = _score(estimator, X_test, y_test.flatten(),
                                                     scorer, error_score)
        score_time = time.time() - start_time - fit_time

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            #if isinstance(test_scores, dict):
            for scorer_name in sorted(test_scores):
                result_msg += f" {scorer_name}: ("
                result_msg += f"test={test_scores})"  # FIXME
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result.update(test_scores)

    if return_n_test_samples:
        for ttc_name, (X_test, _) in test_splits.items():
            result[f"n_{ttc_name}_samples"] = tuple(x.shape[0] for x in X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


# def _score(estimator, X_test, y_test, scorer, error_score="raise"):
#     """Compute the score(s) of an estimator on a given test set.
# 
#     Will return a dict of floats if `scorer` is a dict, otherwise a single
#     float is returned.
#     """
#     if isinstance(scorer, dict):
#         # will cache method calls if needed. scorer() returns a dict
#         scorer = _MultimetricScorer(**scorer)
# 
#     try:
#         if y_test is None:
#             scores = scorer(estimator, X_test)
#         else:
#             scores = scorer(estimator, X_test, y_test)
#     except Exception:
#         if error_score == "raise":
#             raise
#         else:
#             if isinstance(scorer, _MultimetricScorer):
#                 scores = {name: error_score for name in scorer._scorers}
#             else:
#                 scores = error_score
#             warnings.warn(
#                 "Scoring failed. The score on this train-test partition for "
#                 f"these parameters will be set to {error_score}. Details: \n"
#                 f"{format_exc()}",
#                 UserWarning,
#             )
# 
#     error_msg = "scoring must return a number, got %s (%s) instead. (scorer=%s)"
#     if isinstance(scores, dict):
#         for name, score in scores.items():
#             if hasattr(score, "item"):
#                 with suppress(ValueError):
#                     # e.g. unwrap memmapped scalars
#                     score = score.item()
#             if not isinstance(score, numbers.Number):
#                 raise ValueError(error_msg % (score, type(score), name))
#             scores[name] = score
#     else:  # scalar
#         if hasattr(scores, "item"):
#             with suppress(ValueError):
#                 # e.g. unwrap memmapped scalars
#                 scores = scores.item()
#         if not isinstance(scores, numbers.Number):
#             raise ValueError(error_msg % (scores, type(scores), scorer))
#     return scores


def _check_train_test_combinations(
        ttc, ndim, return_train_scores, symbols='LT', sep=''):
    """Check train_test_combinations parameter.
    """
    # ttc = train_test_combinations
    names_provided = ttc is not None and (
        isinstance(ttc[0], str) or isinstance(ttc[0][0], str))

    if ttc is None:
        ret_ttc = itertools.product((0, 1), repeat=ndim)
        if not return_train_scores:
            next(ret_ttc)  # Skip (0, 0, ...), only training data.
        ret_ttc = list(ret_ttc)

    elif any(len(c) != 2 for c in ttc):
        raise ValueError("train_test_combinations must contain only 2-lengthed"
                         "sequences.")

    # If ttc contains string or list-like of strings, translate.
    elif names_provided:
        names = ttc
        ret_ttc = [
            [symbols.index(s) for s in train_test]
            for train_test in names
        ]

    if not names_provided:
        names = [[symbols[is_test]
                     for is_test in train_test]
                 for train_test in ret_ttc]

    names = [n[0]+sep+n[1] for n in names]
    return ret_ttc, names

# NOTE: Originally in sklearn.utils.metaestimators
def _safe_split_nd(estimator, X, y, indices, train_indices=None):
    """Create subset of n-dimensional dataset.

    Slice X, y according to indices for n-dimensional cross-validation.

    .. versionchanged:: hypertree
        No ``estimator`` or ``train_indices`` parameters are necessary by now.

    Parameters
    ----------
    X : list-like of array-like, sparse matrix or iterable
        The feature matrices for each axis.

    y : array-like, sparse matrix or iterable
        Targets to be indexed.

    indices : iterator of array of int
        Slice to select from X and y.

    Returns
    -------
    X_subset : list of array-like, sparse matrix or list
        Indexed data.

    y_subset : array-like, sparse matrix or list
        Indexed targets.

    """
    n_dim = y.ndim
    # TODO: further checking in another function may be adeuqate.
    # X_subset = _safe_indexing(X, indices)
    if not (len(X) == len(indices) == n_dim):
        raise ValueError("Incompatible dimensions. One must ensure "
                         "len(X) == len(indices) == y.ndim")

    if _safe_tags(estimator, key="pairwise"):
        X_subset = []
        for i in range(n_dim):
            Xi, ind = X[i], indices[i]
            if not hasattr(Xi, "shape"):
                raise ValueError(
                    "Precomputed kernels or affinity matrices have "
                    "to be passed as arrays or sparse matrices."
                )
            # Xi is a precomputed square kernel matrix
            if Xi.shape[0] != Xi.shape[1]:
                raise ValueError("Xi should be a square kernel matrix")
            if train_indices is None:
                X_subset.append(Xi[np.ix_(ind, ind)])
            else:
                X_subset.append(Xi[np.ix_(ind, train_indices[i])])
    else:
        X_subset = [Xax[i] for Xax, i in zip(X, indices)]

    if y is not None:
        y_subset = y[np.ix_(*indices)]
    else:
        y_subset = None

    return X_subset, y_subset


# NOTE: Originally in sklearn.utils.__init__
# FIXME: unused by now.
def _safe_indexing_nd(X, indices):
    """Return rows, items or columns of X using indices.

    .. warning::

        This utility is documented, but **private**. This means that
        backward compatibility might be broken without any deprecation
        cycle.

    .. versionchanged:: hypertree
        The original sklearn function has an additional argument to specifying
        the axis of indexing, with it being required to be 0 or 1. Since we are
        in the n-dimensional domain, this restriction is eliminated and the axis
        information must be implicit in the `indices`' format, such as
        (slice(None), [3, 4, 5]) to get the columns 3, 4 and 5 of an ndarray (e-
        quivalent to np.s_[:, [3, 4, 5]]).

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series
        Data from which to sample rows, items or columns. `list` are only
        supported when `axis=0`.
    indices : bool, int, str, slice, array-like

    Returns
    -------
    subset
        Subset of X on axis 0 or 1.

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if indices is None:
        return X

    # TODO: use isinstance.
    # if hasattr(X, "iloc"):
    #     return _pandas_indexing(X, indices, indices_dtype, axis=axis)
    # elif hasattr(X, "shape"):
    #     return _array_indexing(X, indices, indices_dtype, axis=axis)
    if isinstance(X, list):
        return sklearn.utils._list_indexing(X, indices, indices_dtype)
    else:
        return X[indices]
