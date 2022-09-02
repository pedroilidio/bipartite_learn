"""Set of tools to apply standard estimators to bipartite datasets.

TODO: rewrite docs, it's still based on sklearn.pipeline.Pipeline
"""
from __future__ import annotations
from random import random
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import available_if
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from .melter import row_cartesian_product


def melt_Xy(X, y=None, subsample_negatives=False, random_state=None):
    X = row_cartesian_product(X) 

    if y is None:
        return X, y
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must have only 0 or 1 elements.")

    y = y.reshape(-1)

    if subsample_negatives:
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


# TODO: Could simplify code, but currently not using.
def melt_Xy_before(method, subsample_negatives=False, random_state=None):

    def new_method(self, X, y=None, **kwargs):
        Xm, ym = melt_Xy(X, y=y, subsample_negatives=subsample_negatives,
                         random_state=random_state)
        return method(self, X=Xm, y=ym, **kwargs)

    return new_method


# TODO: Could simplify code, but currently not using.
def melt_Xy_before_use_self(method):

    def new_method(self, X, y=None, **kwargs):
        Xm, ym = melt_Xy(X, y=y, subsample_negatives=self.subsample_negatives,
                         random_state=self.random_state)
        return method(self, X=Xm, y=ym, **kwargs)

    return new_method
     

def _estimator_has(attr):
    """Check that estimator has `attr`.

    Used together with `avaliable_if` in `Pipeline` and wrappers.
    """
    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self.estimator, attr)
        return True

    return check


class PU_WrapperND(BaseEstimator):
    def __init__(
        self,
        estimator: BaseEstimator,
        random_state: int|np.random.RandomState|None = None,
        subsample_negatives: bool = False,
        **estimator_args,
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
        if isinstance(estimator, type):
            estimator = estimator(**estimator_args)

        self.estimator = estimator
        self.random_state = random_state
        self.subsample_negatives = subsample_negatives

    def fit(self, X, y=None, **fit_params):
        Xt, yt = melt_Xy(X, y=y, subsample_negatives=self.subsample_negatives,
                         random_state=self.random_state)
        self.estimator.fit(X=Xt, y=yt, **fit_params)
        return self

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        if isinstance(X, list):  # TODO: better way to decide.
            X, _ = melt_Xy(X, y=None, subsample_negatives=False)
        return self.estimator.predict(X)

    @available_if(_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        Xt, yt = melt_Xy(X, y=y, subsample_negatives=self.subsample_negatives,
                        random_state=self.random_state)
        return self.estimator.fit_predict(Xt, yt, **fit_params)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        return self.estimator.predict_proba(X, **predict_proba_params)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        return self.estimator.decision_function(X)

    @available_if(_estimator_has("score_samples"))
    def score_samples(self, X):
        return self.estimator.score_samples(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **predict_log_proba_params):
        return self.estimator.predict_log_proba(X, **predict_log_proba_params)

    @available_if(_estimator_has("transform"))
    def transform(self, X):
        Xt, _ = melt_Xy(X, y=None, subsample_negatives=self.subsample_negatives,
                        random_state=self.random_state)
        return self.transform(Xt)

    @available_if(_estimator_has("fit_transform"))
    def fit_transform(self, X, y=None, **fit_params):
        Xt, yt = melt_Xy(X, y=y, subsample_negatives=self.subsample_negatives,
                        random_state=self.random_state)
        return self.estimator.fit_transform(Xt, yt, **fit_params)
        
    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, Xt):
        X, _ = melt_Xy(Xt, y=None,
                       subsample_negatives=self.subsample_negatives,
                       random_state=self.random_state)
        return self.estimator.inverse_transform(Xt)

    @available_if(_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None):
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


# Metaclass black wizardry method.
class ClassPUWrapperND(type):
    """Patches an sklearn class to receive ND data."""

    def __new__(cls, estimator_class):
        new_name = "ND__" + estimator_class.__name__
        attrs = dict(vars(estimator_class))
        old_init = attrs["__init__"]

        def new_init(self, *, subsample_negatives=False,
                     random_state=None, **kwargs):
            self.subsample_negatives = subsample_negatives
            self.random_state = random_state
            return old_init(self, **kwargs)

        attrs["__init__"] = new_init
        methods_to_patch = ("fit", "transform", "fit_transform",
                            "inverse_transform")
        
        for method in methods_to_patch:
            if method in attrs:
                attrs[method] = melt_Xy_before_use_self(attrs[method])
        
        return type(
            new_name,
            (estimator_class, *estimator_class.__bases__),
            attrs,
        ) 
