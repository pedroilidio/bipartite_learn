"""Set of tools to apply standard estimators to bipartite datasets.

TODO: rewrite docs, it's still based on sklearn.pipeline.Pipeline
"""
from __future__ import annotations
from random import random
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import _BaseComposition, available_if
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from .melter import row_cartesian_product


def _estimator_has(attr):
    """Check that estimator has `attr`.

    Used together with `avaliable_if` in `Pipeline` and wrappers.
    """
    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self.estimator, attr)
        return True

    return check


class PU_WrapperND(_BaseComposition):
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
    
    def _melt_Xy(self, X, y=None):
        X = row_cartesian_product(X) 

        if y is None:
            return X, y
        if not np.isin(y, (0, 1)).all():
            raise ValueError("y must have only 0 or 1 elements.")

        y = y.reshape(-1)

        if self.subsample_negatives:
            random_state = self.random_state or self.estimator.random_state
            random_state = check_random_state(random_state)

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
        Xt, yt = self._melt_Xy(X, y)
        self.estimator.fit(X=Xt, y=yt, **fit_params)
        return self

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Returns the parameters given in the constructor as well as the
        estimators contained within the `steps` of the `Pipeline`.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("estimator", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``. Note that
        you can directly set the parameters of the estimators contained in
        `steps`.
        Parameters
        ----------
        **kwargs : dict
            Parameters of this estimator or parameters of estimators contained
            in `steps`. Parameters of the steps may be set using its name and
            the parameter name separated by a '__'.
        Returns
        -------
        self : object
            Pipeline class instance.
        """
        self._set_params("estimator", **kwargs)
        return self
    
    @available_if(_estimator_has("predict"))  # Will always have.
    def predict(self, X, **predict_params):
        """
        Returns
        -------
        y_pred : ndarray
            Result of calling `predict` on the estimator.
        """
        return self.estimator.predict(X, **predict_params)

    @available_if(_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        """
        Returns
        -------
        y_pred : ndarray
            Result of calling `fit_predict` on the estimator.
        """
        Xt, yt = self._melt_Xy(X, y)
        return self.fit_predict(Xt, yt, **fit_params)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        """
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_proba` on the estimator.
        """
        return self.estimator.predict_proba(X, **predict_proba_params)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        y_score : ndarray of shape (n_samples, n_classes)
            Result of calling `decision_function` on the final estimator.
        """
        return self.estimator.decision_function(X)

    @available_if(_estimator_has("score_samples"))
    def score_samples(self, X):
        """
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            Result of calling `score_samples` on the final estimator.
        """
        return self.esetimator.score_samples(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **predict_log_proba_params):
        """
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        **predict_log_proba_params : dict of string -> object
            Parameters to the ``predict_log_proba`` called at the end of all
            transformations in the pipeline.
        Returns
        -------
        y_log_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_log_proba` on the final estimator.
        """
        return self.estimator.predict_log_proba(X, **predict_log_proba_params)

    @available_if(_estimator_has("transform"))
    def transform(self, X):
        """
        Parameters
        ----------
        X : iterable
        Returns
        -------
        X : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        Xt, _ = self._melt_Xy(X, y=None)
        return self.transform(Xt)

    @available_if(_estimator_has("fit_transform"))
    def fit_transform(self, X, y=None, **fit_params):
        """
        Parameters
        ----------
        X : iterable
        Returns
        -------
        X : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        Xt, yt = self._melt_Xy(X, y)
        return self.estimator.fit_transform(Xt, yt, **fit_params)
        
    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, Xt):
        """Apply `inverse_transform` for each step in a reverse order.
        Parameters
        ----------
        Xt : array-like of shape (n_samples, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_transform`` method.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Inverse transformed data, that is, data in the original feature
            space.
        """
        X, _ = self._melt_Xy(Xt, y=None)
        return self.estimator.inverse_transform(X)

    @available_if(_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None):
        """
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.
        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.
        Returns
        -------
        score : float
            Result of calling `score` on the final estimator.
        """
        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight
        return self.estimator.score(X, y, **score_params)

    @property
    def classes_(self):
        """The classes labels. Only exist if the estimator is a classifier."""
        return self.estimator.classes_

    def _more_tags(self):
        # check if estimator expects pairwise input
        return {"pairwise": _safe_tags(self.estimator, "pairwise")}

    @property
    def n_features_in_(self):
        """Number of features seen during first step `fit` method."""
        return self.estimator.n_features_in_

    @property
    def feature_names_in_(self):
        """Names of features seen during first step `fit` method."""
        # delegate to first step (which will call _check_is_fitted)
        return self.estimator.feature_names_in_

    def __sklearn_is_fitted__(self):
        """Indicate whether the estimator has been fit."""
        try:
            check_is_fitted(self.estimator)
            return True
        except NotFittedError:
            return False