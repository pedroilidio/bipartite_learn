"""Set of tools to apply standard estimators to bipartite datasets.

TODO: Docs.
TODO: check fit inputs.
"""
from __future__ import annotations
from typing import Callable, Sequence
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from .melter import melt_multipartite_dataset
from .utils import check_multipartite_params
from .base import BaseMultipartiteEstimator, BaseMultipartiteSampler
from .utils import _X_is_multipartite


def _estimator_has(attr):
    """Check that primary estimators has `attr`.

    Used together with `avaliable_if` in `GlobalSingleOutputWrapper`.
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
        getattr(self.secondary_rows_estimator_, attr)
        getattr(self.secondary_cols_estimator_, attr)
        return True

    return check


def _primary_estimators_have(attr):
    """Check that secondary estimators have `attr`.

    Used together with `avaliable_if` in `BipartiteLocalWrapper`.
    """
    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self.primary_rows_estimator_, attr)
        getattr(self.primary_cols_estimator_, attr)
        return True

    return check


class IncompatibleEstimatorsError(ValueError):
    """Raised when user tries to wrap incompatible estimators."""


class GlobalSingleOutputWrapper(BaseMultipartiteEstimator, MetaEstimatorMixin):
    def __init__(
        self,
        estimator: BaseEstimator,
        random_state: int | np.random.RandomState | None = None,
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

        Since all multipartite sets are considered jointly, not consecutively,
        and only single output estimators are compatible, this adaptation
        format is called Global Single Output (GSO) [Pliakos _et al._, 2020].

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

    def _melt_input(self, X, y=None):
        if not _X_is_multipartite(X):
            return X, y

        return melt_multipartite_dataset(
            X,
            y,
            subsample_negatives=self.subsample_negatives,
            random_state=self.random_state_,
        )

    def fit(self, X, y=None, **fit_params):
        self.random_state_ = check_random_state(self.random_state)
        Xt, yt = self._melt_input(X, y=y)
        self.estimator = clone(self.estimator)
        self.estimator.fit(X=Xt, y=yt, **fit_params)
        return self

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        X, _ = self._melt_input(X)
        return self.estimator.predict(X)

    @available_if(_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        Xt, yt = self._melt_input(X, y=y)
        return self.estimator.fit_predict(Xt, yt, **fit_params)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        X, _ = self._melt_input(X)
        return self.estimator.predict_proba(X, **predict_proba_params)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        X, _ = self._melt_input(X)
        return self.estimator.decision_function(X)

    @available_if(_estimator_has("score_samples"))
    def score_samples(self, X):
        X, _ = self._melt_input(X)
        return self.estimator.score_samples(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **predict_log_proba_params):
        X, _ = self._melt_input(X)
        return self.estimator.predict_log_proba(X, **predict_log_proba_params)

    @available_if(_estimator_has("transform"))
    def transform(self, X):
        Xt, _ = self._melt_input(X)
        return self.transform(Xt)

    @available_if(_estimator_has("fit_transform"))
    def fit_transform(self, X, y=None, **fit_params):
        Xt, yt = self._melt_input(X, y=y)
        return self.estimator.fit_transform(Xt, yt, **fit_params)
        
    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, Xt):
        X, _ = self._melt_input(Xt)
        return self.estimator.inverse_transform(X)

    @available_if(_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None):
        X, y = self._melt_input(X, y=y)
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


# TODO: docs.
class BipartiteLocalWrapper(BaseMultipartiteEstimator):
    def __init__(
        self,
        primary_estimator: BaseEstimator | Sequence[BaseEstimator],
        secondary_estimator: BaseEstimator | Sequence[BaseEstimator],
        combine_predictions_func: str | Callable = np.mean,
        combine_func_kwargs: dict | None = None,
        independent_labels: bool = True,
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
        self.primary_estimator = primary_estimator
        self.secondary_estimator = secondary_estimator
        self.independent_labels = independent_labels
        self.combine_predictions_func = combine_predictions_func
        self.combine_func_kwargs = combine_func_kwargs

        # FIXME: no validation in init, but some properties depend on this.
        # FIXME: messes up check_is_fitted
        self._check_estimators()  
    
    def _check_estimators(self):
        primary_estimators, secondary_estimators = check_multipartite_params(
            self.primary_estimator, self.secondary_estimator
        )

        for estimator in (*primary_estimators, *secondary_estimators):
            # TODO: 'multioutput_only' tag should imply 'multioutput' tag
            if not (
                _safe_tags(estimator, "multioutput") or
                _safe_tags(estimator, "multioutput_only")
            ):
                raise IncompatibleEstimatorsError(
                    f"All estimators wrapped by {self.__class__.__name__} "
                    f"must support multioutput functionality but {estimator} "
                    "does not. Some meta-estimators defined in scikit-learn "
                    "may be useful, check https://scikit-learn.org/stable/"
                    "modules/multiclass.html"
                )

        self.primary_rows_estimator_ = clone(primary_estimators[0])
        self.primary_cols_estimator_ = clone(primary_estimators[1])
        self.secondary_rows_estimator_ = clone(secondary_estimators[0])
        self.secondary_cols_estimator_ = clone(secondary_estimators[1])

        if (
            getattr(self.secondary_rows_estimator_, "_estimator_type") !=
            getattr(self.secondary_cols_estimator_, "_estimator_type")
        ):
            raise IncompatibleEstimatorsError(
                "Secondary estimators must be of the same type (regressor"
                ", classifier, etc.). See https://scikit-learn.org/stable/"
                "developers/develop.html#estimator-types"
            )

        if (
            _safe_tags(self.primary_rows_estimator_, "pairwise") !=
            _safe_tags(self.primary_cols_estimator_, "pairwise")
        ):
            raise IncompatibleEstimatorsError(
                "Both or none of the primary estimators must be pairwise."
            )

    # TODO: organize 'fit_params'
    def fit(self, X, y=None, **fit_params):
        # TODO: check input
        # self._check_estimators()  # FIXME: remove from init.
        self.X_fit_ = X

        if not self.independent_labels:
            self.y_fit_ = y

        self.primary_rows_estimator_.fit(X=X[0], y=y, **fit_params)
        self.primary_cols_estimator_.fit(X=X[1], y=y.T, **fit_params)

        return self

    def _secondary_fit(self, X):
        """Fit secondary estimators at prediction time."""
        check_is_fitted(self)

        # Transposed because they will be used as training labels.
        new_y_rows = self.primary_rows_estimator_.predict(X[0]).T
        new_y_cols = self.primary_cols_estimator_.predict(X[1]).T

        # If the secondary estimators are able to take advantage of correlated
        # labels, we train them with all the labels we have, even if most pre-
        # dicted columns will be discarded. 
        if not self.independent_labels:
            new_y_cols = np.hstack((new_y_cols, self.y_fit_))
            new_y_rows = np.hstack((new_y_rows, self.y_fit_.T))

        self.secondary_rows_estimator_.fit(self.X_fit_[0], new_y_cols)
        self.secondary_cols_estimator_.fit(self.X_fit_[1], new_y_rows)

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

        rows_pred = self.secondary_rows_estimator_.predict(X[0])
        cols_pred = self.secondary_cols_estimator_.predict(X[1]).T

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
        # self._check_estimators()  # FIXME: remove from init.
        self.X_fit_ = X
        self.primary_rows_estimator_ = clone(self.estimator_rows)
        self.primary_cols_estimator_ = clone(self.estimator_cols)
        self.secondary_rows_estimator_ = clone(self.secondary_estimator_rows)
        self.secondary_cols_estimator_ = clone(self.secondary_estimator_cols)

        if not self.independent_labels:
            self.y_fit_ = y
        
        # Transposed because they will be used as training labels.
        if hasattr(self.primary_rows_estimator_, "fit_predict"):
            new_y_rows = self.primary_rows_estimator_.fit_predict(
                X[0], y, **fit_params).T
            new_y_cols = self.primary_cols_estimator_.fit_predict(
                X[1], y.T, **fit_params).T
        else:
            self.fit(X, y, **fit_params)
            new_y_rows = self.primary_rows_estimator_.predict(X[0]).T
            new_y_cols = self.primary_cols_estimator_.predict(X[1]).T

        # If the secondary estimators are able to take advantage of correlated
        # labels, we train them with all the labels we have, even if most pre-
        # dicted columns will be discarded. 
        if not self.independent_labels:
            new_y_cols = np.hstack((new_y_cols, y))
            new_y_rows = np.hstack((new_y_rows, y.T))

        if hasattr(self.secondary_rows_estimator_, "fit_predict"):
            rows_pred = self.secondary_rows_estimator_.fit_predict(
                X[0], new_y_cols, **fit_params)
            cols_pred = self.secondary_cols_estimator_.fit_predict(
                X[1], new_y_rows, **fit_params).T
        else:
            self.secondary_rows_estimator_.fit(X[0], new_y_cols, **fit_params)
            self.secondary_cols_estimator_.fit(X[1], new_y_rows, **fit_params)
            rows_pred = self.secondary_rows_estimator_.predict(X[0])
            cols_pred = self.secondary_cols_estimator_.predict(X[1]).T

        if not self.independent_labels:
            # Get only the predictions corresponding to the instances being
            # predicted.
            rows_pred = rows_pred[:, :X[1].shape[0]]
            cols_pred = cols_pred[:X[0].shape[0]]

        return self._combine_predictions(rows_pred, cols_pred)

    @available_if(_secondary_estimators_have("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        self._secondary_fit(X)

        rows_pred = self.secondary_rows_estimator_.predict_proba(
            X[0], **predict_proba_params)
        cols_pred = self.secondary_cols_estimator_.predict_proba(
            X[1], **predict_proba_params).T

        if not self.independent_labels:
            # Get only the predictions corresponding to the instances being
            # predicted.
            rows_pred = rows_pred[:, :X[1].shape[0]]
            cols_pred = cols_pred[:X[0].shape[0]]

        # FIXME: Does not work in some cases, such as 'max()'
        return self._combine_predictions(rows_pred, cols_pred)

    @available_if(_secondary_estimators_have("predict_log_proba"))
    def predict_log_proba(self, X, **predict_log_proba_params):
        self._secondary_fit(X)

        rows_pred = self.secondary_rows_estimator_.predict_log_proba(
            X[0], **predict_log_proba_params)
        cols_pred = self.secondary_cols_estimator_.predict_log_proba(
            X[1], **predict_log_proba_params).T

        if not self.independent_labels:
            # Get only the predictions corresponding to the instances being
            # predicted.
            rows_pred = rows_pred[:, :X[1].shape[0]]
            cols_pred = cols_pred[:X[0].shape[0]]

        # FIXME: Does not work in some cases, such as 'max()'
        return self._combine_predictions(rows_pred, cols_pred)

    @property
    def _estimator_type(self):
        return getattr(self.secondary_rows_estimator_, "_estimator_type", None)

    @property
    def classes_(self):
        """The classes labels. Only exist if the estimator is a classifier."""
        return self.estimator.classes_

    def _more_tags(self):
        # check if the primary estimator expects pairwise input
        return {"pairwise": _safe_tags(self.primary_rows_estimator_, "pairwise")}

    @property
    def n_features_in_(self):
        """Number of features seen during first step `fit` method."""
        return (
            self.primary_rows_estimator_.n_features_in_ +
            self.primary_cols_estimator_.n_features_in_
        )

    @property
    def feature_names_in_(self):
        """Names of features seen during first step `fit` method."""
        return (
            self.primary_rows_estimator_.feature_names_in_ +
            self.primary_cols_estimator_.feature_names_in_
        )

    def __sklearn_is_fitted__(self):
        """Indicate whether the primary estimators have been fit."""
        try:
            check_is_fitted(self.primary_rows_estimator_)
            check_is_fitted(self.primary_cols_estimator_)
            return True
        except NotFittedError:
            # Let the caller check_is_fitted() raise the error
            return False


# FIXME: test get_params()
class MultipartiteTransformerWrapper(BaseMultipartiteEstimator,
                                     TransformerMixin):
    """Manages a transformer for each feature space in multipartite datasets.
    """
    def __init__(
        self,
        transformers: BaseEstimator | Sequence[BaseEstimator],
        ndim: int | None = 2,
    ):
        self.transformers = transformers
        self.ndim = ndim

    def fit(self, X, y=None):
        self._set_transformers()
        self._validate_data(X, y)
        for Xi, yi, transformer in self._roll_axes(X, y):
            transformer.fit(Xi, yi)

    def transform(self, X, y=None):
        self._validate_data(X, y)
        return [
            transformer.transform(Xi)
            for Xi, _, transformer in self._roll_axes(X)
        ]

    def fit_transform(self, X, y=None):
        self._set_transformers()
        self._validate_data(X, y)
        return [
            transformer.fit_transform(Xi, yi)
            for Xi, yi, transformer in self._roll_axes(X, y)
        ]

    def _set_transformers(self):
        """Sets self.transformers_ and self.ndim_ during fit.
        """
        if isinstance(self.transformers, Sequence):
            if self.ndim is None:
                self.ndim_ = len(self.transformers)

            elif self.ndim != len(self.transformers):
                raise ValueError("'self.ndim' must correspond to the number of"
                                 " transformers in self.transformers")
            else:
                self.ndim_ = self.ndim

            self.transformers_ = [clone(t) for t in self.transformers]

        else:  # If a single transformer provided
            if self.ndim is None:
                raise ValueError("'ndim' must be provided if 'transformers' is"
                                 " a single transformer.")
            self.ndim_ = self.ndim
            self.transformers_ = [
                clone(self.transformers) for _ in range(self.ndim_)
            ]

    def _validate_data(self, X, y):
        if len(X) != self.ndim_:
            raise ValueError(f"Wrong dimension. X has {len(X)} items, was exp"
                             "ecting {self.ndim}.")
        # FIXME: validate
        # super()._validate_data(X, y, validate_separately=True)

    def _roll_axes(self, X, y=None):
        if not hasattr(self, "transformers_"):
            raise AttributeError("One must call _set_transformers() before"
                                 "attempting to call _roll_axes()")

        for ax in range(self.ndim_):
            yield X[ax], y, self.transformers_[ax]

            if y is not None:
                y = np.moveaxis(y, 0, -1)

    def __sklearn_is_fitted__(self):
        """Indicate whether the transformers have been fit."""
        if not hasattr(self, "ndim_"):
            return False
        try:
            map(check_is_fitted, self.transformers_)
        except NotFittedError:
            # Let the caller check_is_fitted() raise the error
            return False


# TODO: sampler wrapper and transformer wrapper could share code
class MultipartiteSamplerWrapper(BaseMultipartiteSampler):
    """Manages a sampler for each feature space in multipartite datasets.
    """
    def __init__(
        self,
        samplers: BaseEstimator | Sequence[BaseEstimator],
        ndim: int | None = 2,
    ):
        self.samplers = samplers
        self.ndim = ndim

    def _fit_resample(self, X, y):
        # NOTE: ressampled y is discarded!
        self._set_samplers()
        self._validate_data(X, y)
        return (
            [
                # FIXME: imblearn cannot deal with multi-column y
                # sampler.fit_resample(Xi, yi)[0]  # skip validation
                sampler._fit_resample(Xi, yi)[0]
                for Xi, yi, sampler in self._roll_axes(X, y)
            ],
            y,
        )

    def _set_samplers(self):
        """Sets self.samplers_ and self.ndim_ during fit.
        """
        if isinstance(self.samplers, Sequence):
            if self.ndim is None:
                self.ndim_ = len(self.samplers)

            elif self.ndim != len(self.samplers):
                raise ValueError("'self.ndim' must correspond to the number of"
                                 " samplers in self.samplers")
            else:
                self.ndim_ = self.ndim

            self.samplers_ = [clone(t) for t in self.samplers]

        else:  # If a single sampler provided
            if self.ndim is None:
                raise ValueError("'ndim' must be provided if 'samplers' is"
                                 " a single sampler.")
            self.ndim_ = self.ndim
            self.samplers_ = [
                clone(self.samplers) for _ in range(self.ndim_)
            ]

    def _validate_data(self, X, y):
        if len(X) != self.ndim_:
            raise ValueError(f"Wrong dimension. X has {len(X)} items, was exp"
                             "ecting {self.ndim}.")
        # FIXME: validate
        # super()._validate_data(X, y, validate_separately=True)

    def _roll_axes(self, X, y=None):
        if not hasattr(self, "samplers_"):
            raise AttributeError("One must call _set_samplers() before"
                                 "attempting to call _roll_axes()")

        for ax in range(self.ndim_):
            yield X[ax], y, self.samplers_[ax]

            if y is not None:
                y = np.moveaxis(y, 0, -1)
