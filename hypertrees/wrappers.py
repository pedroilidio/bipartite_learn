"""Set of tools to apply standard estimators to bipartite datasets.

TODO: Docs.
TODO: check fit inputs.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils.metaestimators import available_if
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from .melter import row_cartesian_product


def _estimator_has(attr):
    """Check that primary estimators has `attr`.

    Used together with `avaliable_if` in `PU_WrapperND`.
    """
    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self.estimator, attr)
        return True

    return check
     

def _secondary_estimators_have(attr):
    """Check that primary estimators has `attr`.

    Used together with `avaliable_if` in `BipartiteLocalWrapper`.
    """
    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self.secondary_estimator_rows, attr)
        getattr(self.secondary_estimator_cols, attr)
        return True

    return check


def _primary_estimators_have(attr):
    """Check that secondary estimators have `attr`.

    Used together with `avaliable_if` in `BipartiteLocalWrapper`.
    """
    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self.estimator_rows, attr)
        getattr(self.estimator_cols, attr)
        return True

    return check


# class GSOWrapper:
# class GlobalSingleOutputWrapper:
class PU_WrapperND(BaseEstimator, MetaEstimatorMixin):
    def __init__(
        self,
        estimator: BaseEstimator,
        random_state: int|np.random.RandomState|None = None,
        subsample_negatives: bool = False,
    ):
        """Wraps a standard estimator/transformer to work on PU n-partite data.

        With n-partite interaction data, X is actualy a list of sample
        attribute matrices, one for each group of samples (drugs and targets,
        for instance). y is then an interaction matrix (or tensor) denoting
        interaction between the instances of each group.

        We are currently assuming positive-unlabeled data, in which y == 1
        denotes positive and known interaction and y == 0 denotes unknown
        (positive or negative).

        This wrapper melts the n-partite X/y data before passing them to the
        estimator, so that monopartite estimators (standard ones) can be used
        with n-partite cross-validators and pipelines.

        Parameters
        ----------

        estimator : sklearn estimator or transformer
            Estimator/transformed to receive processed data.

        random_state : int or np.random.RandomState or None
            Seed or random state object to be used for negative data
            subsampling. If None, it takes its value from self.estimator.
            Has no effect if subsample_negatives == False.

        subsample_negatives : boolean
            Indicates wether to use all unknown data available (False) or to
            randomly subsample negative pairs (assumed zero-labeled, y == 0)
            to have a balanced dataset.
        """
        self.estimator = estimator
        self.random_state = random_state
        self.subsample_negatives = subsample_negatives

    def melt_Xy(self, X, y=None):
        """Melt bipartite input.
        
        If X is a list of Xi feature matrices, one for each bipartite group,
        convert it to traditional data format by generating concatenations of
        rows from X[0] with rows from X[1].
        """
        if not isinstance(X, list):  # TODO: better way to decide.
            return X, y  # Already molten input.

        X = row_cartesian_product(X) 

        if y is None:
            return X, y
        if not np.isin(y, (0, 1)).all():
            raise ValueError("y must have only 0 or 1 elements.")

        y = y.reshape(-1)

        if self.subsample_negatives:
            random_state = check_random_state(self.random_state)

            mask = (y == 1)
            n_positives = mask.sum()
            prob = ~mask
            prob = prob/prob.sum()

            zeros_to_keep = random_state.choice(
                y.size, size=n_positives, replace=False, p=prob)
            
            mask[zeros_to_keep] = True
            X, y = X[mask], y[mask]
        
        return X, y

    def fit(self, X, y=None, **fit_params):
        Xt, yt = self.melt_Xy(X, y=y)
        self.estimator.fit(X=Xt, y=yt, **fit_params)
        return self

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        X, _ = self.melt_Xy(X, y=None)
        return self.estimator.predict(X)

    @available_if(_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        Xt, yt = self.melt_Xy(X, y=y)
        return self.estimator.fit_predict(Xt, yt, **fit_params)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        X, _ = self.melt_Xy(X, y=None)
        return self.estimator.predict_proba(X, **predict_proba_params)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        X, _ = self.melt_Xy(X, y=None)
        return self.estimator.decision_function(X)

    @available_if(_estimator_has("score_samples"))
    def score_samples(self, X):
        X, _ = self.melt_Xy(X, y=None)
        return self.estimator.score_samples(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **predict_log_proba_params):
        X, _ = self.melt_Xy(X, y=None)
        return self.estimator.predict_log_proba(X, **predict_log_proba_params)

    @available_if(_estimator_has("transform"))
    def transform(self, X):
        Xt, _ = self.melt_Xy(X, y=None)
        return self.transform(Xt)

    @available_if(_estimator_has("fit_transform"))
    def fit_transform(self, X, y=None, **fit_params):
        Xt, yt = self.melt_Xy(X, y=y)
        return self.estimator.fit_transform(Xt, yt, **fit_params)
        
    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, Xt):
        X, _ = self.melt_Xy(Xt, y=None)
        return self.estimator.inverse_transform(X)

    @available_if(_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None):
        X, y = self.melt_Xy(X, y=y)
        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight
        return self.estimator.score(X, y, **score_params)

    @property
    def _estimator_type(self):
        return getattr(self.estimator, "_estimator_type", None)

    @property
    def classes_(self):
        """The classes labels. Only exist if the estimator is a classifier."""
        return self.estimator.classes_

    @property
    def n_features_in_(self):
        """Number of features seen during `fit`."""
        return self.estimator.n_features_in_

    @property
    def feature_names_in_(self):
        """Names of features seen during `fit`."""
        return self.estimator.feature_names_in_

    def __sklearn_is_fitted__(self):
        """Indicate whether the estimator has been fit."""
        try:
            check_is_fitted(self.estimator)
            return True
        except NotFittedError:
            return False


class BipartiteLocalWrapper(BaseEstimator):
    def __init__(
        self,
        estimator_rows: BaseEstimator,
        estimator_cols: BaseEstimator,
        secondary_estimator_rows: BaseEstimator,
        secondary_estimator_cols: BaseEstimator,
        combine_predictions_func: str | Callable = np.mean,
        combine_func_kwargs: dict | None = None,
        independent_labels: bool = True,
    ):
        # TODO: organize 'fit_params'
        """Wraps a standard estimator/transformer to work on PU n-partite data.

        With n-partite interaction data, X is actualy a list of sample
        attribute matrices, one for each group of samples (drugs and targets,
        for instance). y is then an interaction matrix (or tensor) denoting
        interaction between the instances of each group.

        We are currently assuming positive-unlabeled data, in which y == 1
        denotes positive and known interaction and y == 0 denotes unknown
        (positive or negative).

        This wrapper melts the n-partite X/y data before passing them to the
        estimator, so that monopartite estimators (standard ones) can be used
        with n-partite cross-validators and pipelines.

        Parameters
        ----------

        estimator : sklearn estimator or transformer
            Estimator/transformed to receive processed data.

        random_state : int or np.random.RandomState or None
            Seed or random state object to be used for negative data
            subsampling. If None, it takes its value from self.estimator.
            Has no effect if subsample_negatives == False.

        subsample_negatives : boolean
            Indicates wether to use all unknown data available (False) or to
            randomly subsample negative pairs (assumed zero-labeled, y == 0)
            to have a balanced dataset.
        """
        self.estimator_rows = estimator_rows
        self.estimator_cols = estimator_cols
        self.secondary_estimator_rows = secondary_estimator_rows
        self.secondary_estimator_cols = secondary_estimator_cols
        self.independent_labels = independent_labels
        self.combine_predictions_func = combine_predictions_func
        self.combine_func_kwargs = combine_func_kwargs

    def fit(self, X, y=None, **fit_params):
        # TODO: check input
        self.X_ = X

        if not self.independent_labels:
            self.y_ = y

        self.estimator_rows.fit(X=X[0], y=y, **fit_params)
        self.estimator_cols.fit(X=X[1], y=y.T, **fit_params)

        return self

    def _secondary_fit(self, X):
        """Fit secondary estimators at prediction time."""
        check_is_fitted(self)

        # Transposed because they will be used as training labels.
        new_y_rows = self.estimator_rows.predict(X[0]).T
        new_y_cols = self.estimator_cols.predict(X[1]).T

        # If the secondary estimators are able to take advantage of correlated
        # labels, we train them with all the labels we have, even if most pre-
        # dicted columns will be discarded. 
        if not self.independent_labels:
            new_y_cols = np.hstack(new_y_cols, self.y_)
            new_y_rows = np.hstack(new_y_rows, self.y_.T)

        self.secondary_estimator_rows.fit(self.X_[0], new_y_cols)
        self.secondary_estimator_cols.fit(self.X_[1], new_y_rows)

        return self
    
    def _combine_predictions(self, rows_pred, cols_pred):
        return np.apply_along_axis(
            func1d=self.combine_predictions_func,
            axis=0,
            arr=(rows_pred, cols_pred),
            **(self.combine_func_kwargs or {}),
        ).reshape(-1)

    @available_if(_secondary_estimators_have("predict"))
    def predict(self, X):
        self._secondary_fit(X)

        rows_pred = self.secondary_estimator_rows.predict(X[0])
        cols_pred = self.secondary_estimator_cols.predict(X[1]).T

        if not self.independent_labels:
            # Get only the predictions corresponding to the instances being
            # predicted.
            rows_pred = rows_pred[:, :X[1].shape[0]]
            cols_pred = cols_pred[:X[0].shape[0]]

        return self._combine_predictions(rows_pred, cols_pred)
    
    @available_if(lambda self: _primary_estimators_have("fit_predict") or
                               _secondary_estimators_have("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        # TODO: check input
        self.X_ = X

        if not self.independent_labels:
            self.y_ = y
        
        # Transposed because they will be used as training labels.
        if _primary_estimators_have("fit_predict")(self):
            new_y_rows = self.estimator_rows.fit_predict(
                X[0], y, **fit_params).T
            new_y_cols = self.estimator_cols.fit_predict(
                X[1], y.T, **fit_params).T
        else:
            self.fit(X, y, **fit_params)
            new_y_rows = self.estimator_rows.predict(X[0]).T
            new_y_cols = self.estimator_cols.predict(X[1]).T

        # If the secondary estimators are able to take advantage of correlated
        # labels, we train them with all the labels we have, even if most pre-
        # dicted columns will be discarded. 
        if not self.independent_labels:
            new_y_cols = np.hstack(new_y_cols, y)
            new_y_rows = np.hstack(new_y_rows, y.T)

        if _secondary_estimators_have("fit_predict")(self):
            rows_pred = self.secondary_estimator_rows.fit_predict(
                X[0], new_y_cols, **fit_params)
            cols_pred = self.secondary_estimator_cols.fit_predict(
                X[1], new_y_rows, **fit_params).T
        else:
            self.secondary_estimator_rows.fit(X[0], new_y_cols, **fit_params)
            self.secondary_estimator_cols.fit(X[1], new_y_rows, **fit_params)
            rows_pred = self.secondary_estimator_rows.predict(X[0])
            cols_pred = self.secondary_estimator_cols.predict(X[1]).T

        if not self.independent_labels:
            # Get only the predictions corresponding to the instances being
            # predicted.
            rows_pred = rows_pred[:, :X[1].shape[0]]
            cols_pred = cols_pred[:X[0].shape[0]]

        return self.combine_predictions_func(rows_pred, cols_pred).reshape(-1)

    @available_if(_secondary_estimators_have("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        self._secondary_fit(X)

        rows_pred = self.secondary_estimator_rows.predict_proba(
            X[0], **predict_proba_params)
        cols_pred = self.secondary_estimator_cols.predict_proba(
            X[1], **predict_proba_params).T

        if not self.independent_labels:
            # Get only the predictions corresponding to the instances being
            # predicted.
            rows_pred = rows_pred[:, :X[1].shape[0]]
            cols_pred = cols_pred[:X[0].shape[0]]

        # FIXME: Does not work in some cases, such as 'max()'
        return self.combine_predictions_func(rows_pred, cols_pred).reshape(-1)

    @available_if(_secondary_estimators_have("predict_log_proba"))
    def predict_log_proba(self, X, **predict_log_proba_params):
        self._secondary_fit(X)

        rows_pred = self.secondary_estimator_rows.predict_log_proba(
            X[0], **predict_log_proba_params)
        cols_pred = self.secondary_estimator_cols.predict_log_proba(
            X[1], **predict_log_proba_params).T

        if not self.independent_labels:
            # Get only the predictions corresponding to the instances being
            # predicted.
            rows_pred = rows_pred[:, :X[1].shape[0]]
            cols_pred = cols_pred[:X[0].shape[0]]

        # FIXME: Does not work in some cases, such as 'max()'
        return self.combine_predictions_func(rows_pred, cols_pred).reshape(-1)

    @property
    def _estimator_type(self):
        return getattr(self.estimator, "_estimator_type", None)

    @property
    def classes_(self):
        """The classes labels. Only exist if the estimator is a classifier."""
        return self.estimator.classes_

    def _more_tags(self):
        # check if the primary estimator expects pairwise input
        return {"pairwise": _safe_tags(self.primary_estimator_rows, "pairwise")}

    @property
    def n_features_in_(self):
        """Number of features seen during first step `fit` method."""
        return (
            self.estimator_rows.n_features_in_ +
            self.estimator_cols.n_features_in_
        )

    @property
    def feature_names_in_(self):
        """Names of features seen during first step `fit` method."""
        return (
            self.estimator_rows.feature_names_in_ +
            self.estimator_cols.feature_names_in_
        )

    def __sklearn_is_fitted__(self):
        """Indicate whether the primary estimators have been fit."""
        try:
            check_is_fitted(self.estimator_rows)
            check_is_fitted(self.estimator_cols)
            return True
        except NotFittedError:
            return False
