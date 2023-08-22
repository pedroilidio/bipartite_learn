"""Set of tools to apply standard estimators to bipartite datasets.

TODO: Docs.
TODO: check fit inputs.
"""
from __future__ import annotations
from typing import Callable, Sequence
import numpy as np
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MetaEstimatorMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from imblearn.base import BaseSampler
from .melter import melt_multipartite_dataset
from .utils import check_multipartite_params
from .base import (
    BaseMultipartiteEstimator,
    BaseBipartiteEstimator,
    BaseMultipartiteSampler,
)
from .utils import _X_is_multipartite

__all__ = [
    "GlobalSingleOutputWrapper",
    "LocalMultiOutputWrapper",
    "MultipartiteTransformerWrapper",
    "MultipartiteSamplerWrapper",
]


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

    Used together with `avaliable_if` in `LocalMultiOutputWrapper`.
    """

    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self.secondary_rows_estimator, attr)
        getattr(self.secondary_cols_estimator, attr)
        return True

    return check


def _primary_estimators_have(attr):
    """Check that secondary estimators have `attr`.

    Used together with `avaliable_if` in `LocalMultiOutputWrapper`.
    """

    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self.primary_rows_estimator, attr)
        getattr(self.primary_cols_estimator, attr)
        return True

    return check


class IncompatibleEstimatorsError(ValueError):
    """Raised when user tries to wrap incompatible estimators."""


class GlobalSingleOutputWrapper(BaseMultipartiteEstimator, MetaEstimatorMixin):
    """Employ the GSO strategy to adapt sstandard estimators to bipartite data.

    In this strategy, the estimator is applied to concatenations of a feature
    vector from the first sample domain with a feature vector from the second
    domain, while `y` is considered a unidimensional vector.

    .. image:: _static/user_guide/gso.svg
        :align: center
        :width: 50%
        :class: only-light

    .. image:: _static/user_guide/gso_dark.svg
        :align: center
        :width: 50%
        :class: only-dark

    Read more in the :ref:`User Guide <user_guide:global_single_output>`.

    References
    ----------
    .. [1] :doi:`Global Multi-Output Decision Trees for interaction prediction \
       <doi.org/10.1007/s10994-018-5700-x>`
       Pliakos, Geurts and Vens, 2018
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        under_sampler: BaseSampler | None = None,
    ):
        """Initialize the wrapper.

        Parameters
        ----------
        estimator : BaseEstimator
            Estimator or transformer to receive the adapted bipartite data.

        under_sampler : BaseSampler or None, default=None
            Optional sampler to be applied to the transformed data before
            fitting the estimator. Useful in cases where using all the possible
            interactions is not computationally feasible.
        """
        self.estimator = estimator
        self.under_sampler = under_sampler

    def _melt_input(self, X, y=None):
        if not _X_is_multipartite(X):
            return X, y
        return melt_multipartite_dataset(X, y)

    def _fit_resample_input(self, X, y):
        if self.under_sampler is None:
            return X, y
        self.under_sampler_ = clone(self.under_sampler)
        return self.under_sampler_.fit_resample(X, y)

    def fit(self, X, y=None, **fit_params):
        X, y = self._validate_data(X, y, reset=False)
        Xt, yt = self._melt_input(X, y=y)
        Xt, yt = self._fit_resample_input(Xt, yt)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X=Xt, y=yt, **fit_params)
        return self

    @available_if(_estimator_has("predict"))
    def predict(self, X, **predict_params):
        X, _ = self._melt_input(X)
        return self.estimator_.predict(X, **predict_params)

    @available_if(_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        X, y = self._validate_data(X, y, reset=False)
        Xt, yt = self._melt_input(X, y=y)
        Xt, yt = self._fit_resample_input(Xt, yt)
        self.estimator_ = clone(self.estimator)
        return self.estimator_.fit_predict(Xt, yt, **fit_params)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        X, _ = self._melt_input(X)
        return self.estimator_.predict_proba(X, **predict_proba_params)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        X, _ = self._melt_input(X)
        return self.estimator_.decision_function(X)

    @available_if(_estimator_has("score_samples"))
    def score_samples(self, X):
        X, _ = self._melt_input(X)
        return self.estimator_.score_samples(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **predict_log_proba_params):
        X, _ = self._melt_input(X)
        return self.estimator_.predict_log_proba(X, **predict_log_proba_params)

    @available_if(_estimator_has("transform"))
    def transform(self, X):
        Xt, _ = self._melt_input(X)
        return self.estimator_.transform(Xt)

    @available_if(_estimator_has("fit_transform"))
    def fit_transform(self, X, y=None, **fit_params):
        X, y = self._validate_data(X, y, reset=False)
        Xt, yt = self._melt_input(X, y=y)
        Xt, yt = self._fit_resample_input(Xt, yt)
        self.estimator_ = clone(self.estimator)
        return self.estimator_.fit_transform(Xt, yt, **fit_params)

    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, Xt):
        X, _ = self._melt_input(Xt)
        return self.estimator_.inverse_transform(X)

    @available_if(_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None):
        X, y = self._melt_input(X, y=y)
        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight
        return self.estimator_.score(X, y, **score_params)

    @property
    def _estimator_type(self):
        return getattr(self.estimator, "_estimator_type", None)

    @property
    def classes_(self):
        """The classes labels. Only exist if the estimator is a classifier."""
        return self.estimator_.classes_

    @property
    def n_features_in_(self):
        """Number of features seen during `fit`."""
        return self.estimator_.n_features_in_

    @property
    def feature_names_in_(self):
        """Names of features seen during `fit`."""
        return self.estimator_.feature_names_in_

    def __sklearn_is_fitted__(self):
        """Indicate whether the estimator has been fit."""
        return hasattr(self, "estimator_")


class LocalMultiOutputWrapper(BaseBipartiteEstimator):
    """Implements the Local Multi-Output strategy for adapting estimators.
    
    This wrapper facilitates the implementation of the local multi-output
    approach to adapt monopartite estimators to bipartite scenarios. In this
    approach, four multi-output estimators are aggregated.

    The training procedure (calling ``fit(X_train, y)``) consists simply of:

    1. Train a primary rows estimator on X_train[0] and y_train.
    2. Train a primary columns estimator on X_train[1] and y_train.T.

    The prediction procedure then utilities the predictions of the primary
    estimators in order to be able to make predictions on completely new
    interactions. ``predict(X_test)`` will perform the following steps:

    1. a. Use ``self.primary_cols_estimator_`` to predict new columns for
            the interaction matrix, that correspond to the targets of
            ``X_test[0]``.
    1. b. Use ``self.primary_rows_estimator_`` to predict new rows for the
            interaction matrix, that correspond to the targets of X_test[1].

    2. a. Fit the secondary rows estimator on the newly predicted columns
            and X_test[0].
    2. a. Fit the secondary columns estimator on the newly predicted rows
            and X_test[1].
    
    3. Combine the predictions of the secondary estimators using
        ``self.combine_predictions_func(rows_pred, cols_pred)``.

    If ``self.independent_labels`` is ``False``, then the original
    training data is appended to the training data of the secondary
    estimators in step 2, allowing the secondary estimators to explore
    inter-output correlations. 

    See the :ref:`User Guide <user_guide:local_multi_output>` for a diagram and
    more information.

    Attributes
    ----------
    primary_rows_estimator_ : BaseEstimator
        The fitted primary rows estimator.
    primary_cols_estimator_ : BaseEstimator
        The fitted primary columns estimator.
    secondary_rows_estimator_ : BaseEstimator
        The fitted secondary rows estimator.
    secondary_cols_estimator_ : BaseEstimator
        The fitted secondary columns estimator.

    Notes
    -----
    Note that the secondary estimators must be refit every time the
    wrapper's ``predict()`` method is called, which may increase prediction
    time depending on the type of secondary estimators chosen by the user.
    
    Compositions of single-output estimators can also be used
    instead of multi-output estimators, which can be implemented with
    scikit-learn wrappers such as :class:`MultiOutputRegressor` or
    :class:`MultiOutputClassifier`. This could be an interesting option in
    cases where the base estimators do not natively support multiple
    outputs.
        
    See Also
    --------
    GlobalSingleOutputWrapper : A wrapper that fits a single-output estimator
        to bipartite datasets.
    MultiOutputRegressor : A scikit-learn wrapper that fits a separate regressor
        for each output variable.
    MultiOutputClassifier : A scikit-learn wrapper that fits a separate
        classifier for each output variable.
    
    Examples
    --------

    .. code-block:: python

        from bipartite_learn.datasets import NuclearReceptorsLoader
        from bipartite_learn.wrappers import LocalMultiOutputWrapper
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.multioutput import MultiOutputClassifier
        
        X, y = NuclearReceptorsLoader().load()  # X is a list of two matrices
        bipartite_clf = LocalMultiOutputWrapper(
            primary_rows_estimator=MultiOutputClassifier(SVC()),
            primary_cols_estimator=MultiOutputClassifier(SVC()),
            secondary_rows_estimator=KNeighborsClassifier(),
            secondary_cols_estimator=KNeighborsClassifier(),
        )
        bipartite_clf.fit(X, y)
    
    References
    ----------
    .. [1] :doi:`Global Multi-Output Decision Trees for interaction prediction \
       <doi.org/10.1007/s10994-018-5700-x>`
       Pliakos, Geurts and Vens, 2018
    
    .. [2] :doi:`Yamanishi, Y., Araki, M., Gutteridge, A., Honda, W., &
           Kanehisa, M. (2008). Prediction of drug-target interaction networks
           from the integration of chemical and genomic spaces. Bioinformatics,
           24(13), i232–i240. <https://doi.org/10.1093%2Fbioinformatics%2Fbtn162>`
    """

    def __init__(
        self,
        primary_rows_estimator: BaseEstimator,
        primary_cols_estimator: BaseEstimator,
        secondary_rows_estimator: BaseEstimator,
        secondary_cols_estimator: BaseEstimator,
        combine_predictions_func: Callable = np.mean,
        combine_func_kwargs: dict | None = None,
        independent_labels: bool = True,
    ):
        """Initializes the wrapper.

        This wrapper facilitates the implementation of the local multi-output
        approach to adapt monopartite estimators to bipartite scenarios. In this
        approach, four multi-output estimators are aggregated.

        Parameters
        ----------
        primary_rows_estimator : BaseEstimator
            The estimator to be used as the primary rows estimator.
        primary_cols_estimator : BaseEstimator
            The estimator to be used as the primary columns estimator.
        secondary_rows_estimator : BaseEstimator
            The estimator to be used as the secondary rows estimator.
        secondary_cols_estimator : BaseEstimator
            The estimator to be used as the secondary columns estimator.
        combine_predictions_func : Callable, default=np.mean
            The function used to combine the predictions of the secondary
            estimators.
        combine_func_kwargs : dict, default=None
            A dictionary of keyword arguments to be passed to the
            ``combine_predictions_func`` function. If None, no keyword arguments
            are passed.
        independent_labels : bool, default=True
            If ``True``, the secondary estimators are trained on only the data
            predicted by the primary estimators. If ``False``, the original
            training data is appended to the training data of the secondary
            estimators, allowing the secondary estimators to explore
            inter-output correlations.

        See the :ref:`User Guide <local_multi_output>` for more information on
        the Local Multi-Output strategy for adapting estimators.
        """
        self.primary_rows_estimator = primary_rows_estimator
        self.primary_cols_estimator = primary_cols_estimator
        self.secondary_rows_estimator = secondary_rows_estimator
        self.secondary_cols_estimator = secondary_cols_estimator

        self.independent_labels = independent_labels
        self.combine_predictions_func = combine_predictions_func
        self.combine_func_kwargs = combine_func_kwargs

    def _check_estimators(self):
        for estimator in (
            self.primary_rows_estimator,
            self.primary_cols_estimator,
            self.secondary_rows_estimator,
            self.secondary_cols_estimator,
        ):
            # TODO: 'multioutput_only' tag should imply 'multioutput' tag
            if not (
                _safe_tags(estimator, "multioutput")
                or _safe_tags(estimator, "multioutput_only")
            ):
                raise IncompatibleEstimatorsError(
                    f"All estimators wrapped by {self.__class__.__name__} "
                    f"must support multioutput functionality but {estimator} "
                    "does not. Some meta-estimators defined in scikit-learn "
                    "may be useful, check https://scikit-learn.org/stable/"
                    "modules/multiclass.html"
                )

        if getattr(self.secondary_rows_estimator, "_estimator_type") != getattr(
            self.secondary_cols_estimator, "_estimator_type"
        ):
            raise IncompatibleEstimatorsError(
                "Secondary estimators must be of the same type (regressor"
                ", classifier, etc.). See https://scikit-learn.org/stable/"
                "developers/develop.html#estimator-types"
            )

        if _safe_tags(self.primary_rows_estimator, "pairwise") != _safe_tags(
            self.primary_cols_estimator, "pairwise"
        ):
            raise IncompatibleEstimatorsError(
                "Both or none of the primary estimators must be pairwise."
            )

        self.primary_rows_estimator_ = clone(self.primary_rows_estimator)
        self.primary_cols_estimator_ = clone(self.primary_cols_estimator)
        self.secondary_rows_estimator_ = clone(self.secondary_rows_estimator)
        self.secondary_cols_estimator_ = clone(self.secondary_cols_estimator)

    def fit(self, X, y, **fit_params):
        """Fits the wrapper to the training data.

        Raises
        ------
        IncompatibleEstimatorsError
            If any of the estimators passed as arguments does not support
            multi-output functionality.  If the secondary estimators are not of
            the same type (e.g., regressor, classifier).  If only one of the
            primary estimators is pairwise.
        """
        if not _X_is_multipartite(X):
            # TODO: specific error and function to check multipartite data
            # TODO: set estimator tag 'multipartite_only' to True
            raise ValueError("X must be multipartite.")
        self._check_estimators()
        X, y = self._validate_data(X, y, reset=False)
        self.X_fit_ = X

        if not self.independent_labels:
            self.y_fit_ = y

        self.primary_rows_estimator_.fit(X[0], y, **fit_params)
        self.primary_cols_estimator_.fit(X[1], y.T, **fit_params)

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
        )

    @available_if(_secondary_estimators_have("predict"))
    def predict(self, X, **predict_params):
        if is_classifier(self):
            return self._predict_classification(X, **predict_params)
        elif is_regressor(self):
            return self._predict_regression(X, **predict_params)
        raise ValueError

    def _predict_regression(self, X, **predict_params):
        self._secondary_fit(X)

        rows_pred = self.secondary_rows_estimator_.predict(
            X[0],
            **predict_params,
        )
        cols_pred = self.secondary_cols_estimator_.predict(
            X[1],
            **predict_params,
        ).T

        if not self.independent_labels:
            # Get only the predictions corresponding to the instances being
            # predicted.
            rows_pred = rows_pred[:, : X[1].shape[0]]
            cols_pred = cols_pred[: X[0].shape[0]]

        return self._combine_predictions(rows_pred, cols_pred).reshape(-1)

    def _predict_classification(self, X, **predict_params):
        check_is_fitted(self)

        if hasattr(self, "predict_proba"):
            proba = self.predict_proba(X, **predict_params)
        elif hasattr(self, "predict_log_proba"):
            proba = self.predict_log_proba(X, **predict_params)
        elif hasattr(self, "decision_function"):
            proba = self.decision_function(X, **predict_params)
        else:
            raise ValueError(
                "If a classifier, the estimator must have a predict_proba,"
                "predict_log_proba or decision_function method."
            )

        # FIXME: we must ensure that the classes are the same for all rows
        # and columns of y, and that they are in the same order.
        return self.classes_[np.argmax(proba, axis=1)]

    # TODO: make also available if only a row or column estimator has
    @available_if(
        lambda self: (
            is_classifier(self)
            and (
                _primary_estimators_have("fit_predict")
                or _secondary_estimators_have("fit_predict")
            )
        )
    )
    def fit_predict(self, X, y=None, **fit_params):
        self._check_estimators()
        X, y = self._validate_data(X, y)
        self.X_fit_ = X

        if not self.independent_labels:
            self.y_fit_ = y

        # Transposed because they will be used as training labels.
        if hasattr(self.primary_rows_estimator_, "fit_predict"):
            new_y_rows = self.primary_rows_estimator_.fit_predict(
                X[0],
                y,
                **fit_params,
            ).T
            new_y_cols = self.primary_cols_estimator_.fit_predict(
                X[1],
                y.T,
                **fit_params,
            ).T
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
                X[0], new_y_cols, **fit_params
            )
            cols_pred = self.secondary_cols_estimator_.fit_predict(
                X[1], new_y_rows, **fit_params
            ).T
        else:
            self.secondary_rows_estimator_.fit(X[0], new_y_cols)
            self.secondary_cols_estimator_.fit(X[1], new_y_rows)
            rows_pred = self.secondary_rows_estimator_.predict(X[0])
            cols_pred = self.secondary_cols_estimator_.predict(X[1]).T

        if not self.independent_labels:
            # Get only the predictions corresponding to the instances being
            # predicted.
            rows_pred = rows_pred[:, : X[1].shape[0]]
            cols_pred = cols_pred[: X[0].shape[0]]

        return self._combine_predictions(rows_pred, cols_pred).reshape(-1)

    @available_if(_secondary_estimators_have("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        self._secondary_fit(X)

        rows_pred = np.array(
            self.secondary_rows_estimator_.predict_proba(X[0], **predict_proba_params)
        )
        cols_pred = np.array(
            self.secondary_cols_estimator_.predict_proba(X[1], **predict_proba_params)
        ).transpose((1, 0, 2))

        if not self.independent_labels:
            # Get only the predictions corresponding to the instances being
            # predicted.
            rows_pred = rows_pred[:, : X[1].shape[0]]
            cols_pred = cols_pred[: X[0].shape[0]]

        # FIXME: Does not work in some cases, such as 'max()'
        return self._combine_predictions(rows_pred, cols_pred).reshape(-1, 2)

    @available_if(_secondary_estimators_have("predict_log_proba"))
    def predict_log_proba(self, X, **predict_log_proba_params):
        self._secondary_fit(X)

        rows_pred = np.array(
            self.secondary_rows_estimator_.predict_log_proba(
                X[0], **predict_log_proba_params
            )
        )
        cols_pred = np.array(
            self.secondary_cols_estimator_.predict_log_proba(
                X[1], **predict_log_proba_params
            )
        ).transpose((1, 0, 2))

        if not self.independent_labels:
            # Get only the predictions corresponding to the instances being
            # predicted.
            rows_pred = rows_pred[:, : X[1].shape[0]]
            cols_pred = cols_pred[: X[0].shape[0]]

        # FIXME: Does not work in some cases, such as 'max()'
        return self._combine_predictions(rows_pred, cols_pred).reshape(-1, 2)

    @available_if(_secondary_estimators_have("decision_function"))
    def decision_function(self, X, **decision_function_params):
        self._secondary_fit(X)

        rows_pred = self.secondary_rows_estimator_.decision_function(
            X[0], **decision_function_params
        )
        cols_pred = self.secondary_cols_estimator_.decision_function(
            X[1], **decision_function_params
        ).T

        if not self.independent_labels:
            # Get only the predictions corresponding to the instances being
            # predicted.
            rows_pred = rows_pred[:, : X[1].shape[0]]
            cols_pred = cols_pred[: X[0].shape[0]]

        # FIXME: Does not work in some cases, such as 'max()'
        return self._combine_predictions(rows_pred, cols_pred).reshape(-1)

    @available_if(_secondary_estimators_have("score"))
    def score(self, X, y=None):
        return type(self.secondary_rows_estimator_).score(self, X, y.reshape(-1))

    @property
    def _estimator_type(self):
        return getattr(self.secondary_rows_estimator_, "_estimator_type", None)

    @property
    def classes_(self):
        """The classes labels. Only exist if the estimator is a classifier."""
        # FIXME: The secondary columns estimator generates different classes.
        return np.unique(self.secondary_rows_estimator_.classes_)

    def _more_tags(self):
        # check if the primary estimator expects pairwise input (the primary
        # columns is ensured to be pairwise by the _check_estimators method)
        # TODO: separate into pairwise_rows and pairwise_cols.
        return {"pairwise": _safe_tags(self.primary_rows_estimator, "pairwise")}

    @property
    def n_features_in_(self):
        """Number of features seen during fit."""
        return (
            self.primary_rows_estimator_.n_features_in_
            + self.primary_cols_estimator_.n_features_in_
        )

    @property
    def feature_names_in_(self):
        """Names of features seen during first step `fit` method."""
        return (
            self.primary_rows_estimator_.feature_names_in_
            + self.primary_cols_estimator_.feature_names_in_
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


class MultipartiteTransformerWrapper(
    BaseMultipartiteEstimator,
    TransformerMixin,
):
    """Manages a transformer for each feature space in multipartite datasets."""

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
        return [transformer.transform(Xi) for Xi, _, transformer in self._roll_axes(X)]

    def fit_transform(self, X, y=None):
        self._set_transformers()
        self._validate_data(X, y)
        return [
            transformer.fit_transform(Xi, yi)
            for Xi, yi, transformer in self._roll_axes(X, y)
        ]

    def _set_transformers(self):
        """Sets self.transformers_ and self.ndim_ during fit."""
        if isinstance(self.transformers, Sequence):
            if self.ndim is None:
                self.ndim_ = len(self.transformers)

            elif self.ndim != len(self.transformers):
                raise ValueError(
                    "'self.ndim' must correspond to the number of"
                    " transformers in self.transformers"
                )
            else:
                self.ndim_ = self.ndim

            self.transformers_ = [clone(t) for t in self.transformers]

        else:  # If a single transformer provided
            if self.ndim is None:
                raise ValueError(
                    "'ndim' must be provided if 'transformers' is"
                    " a single transformer."
                )
            self.ndim_ = self.ndim
            self.transformers_ = [clone(self.transformers) for _ in range(self.ndim_)]

    def _validate_data(self, X, y):
        if len(X) != self.ndim_:
            raise ValueError(
                f"Wrong dimension. X has {len(X)} items, was exp" "ecting {self.ndim}."
            )
        # FIXME: validate
        # super()._validate_data(X, y, validate_separately=True)

    def _roll_axes(self, X, y=None):
        if not hasattr(self, "transformers_"):
            raise AttributeError(
                "One must call _set_transformers() before"
                "attempting to call _roll_axes()"
            )

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
    """Manages a sampler for each feature space in multipartite datasets."""

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
        """Sets self.samplers_ and self.ndim_ during fit."""
        if isinstance(self.samplers, Sequence):
            if self.ndim is None:
                self.ndim_ = len(self.samplers)

            elif self.ndim != len(self.samplers):
                raise ValueError(
                    "'self.ndim' must correspond to the number of"
                    " samplers in self.samplers"
                )
            else:
                self.ndim_ = self.ndim

            self.samplers_ = [clone(t) for t in self.samplers]

        else:  # If a single sampler provided
            if self.ndim is None:
                raise ValueError(
                    "'ndim' must be provided if 'samplers' is" " a single sampler."
                )
            self.ndim_ = self.ndim
            self.samplers_ = [clone(self.samplers) for _ in range(self.ndim_)]

    def _validate_data(self, X, y):
        if len(X) != self.ndim_:
            raise ValueError(
                f"Wrong dimension. X has {len(X)} items, was exp" "ecting {self.ndim}."
            )
        # FIXME: validate
        # super()._validate_data(X, y, validate_separately=True)

    def _roll_axes(self, X, y=None):
        if not hasattr(self, "samplers_"):
            raise AttributeError(
                "One must call _set_samplers() before" "attempting to call _roll_axes()"
            )

        for ax in range(self.ndim_):
            yield X[ax], y, self.samplers_[ax]

            if y is not None:
                y = np.moveaxis(y, 0, -1)
