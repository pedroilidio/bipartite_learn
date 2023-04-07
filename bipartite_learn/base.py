import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import (
    check_X_y, check_array, _check_y, _num_features,
)
from imblearn.base import SamplerMixin
from .utils import check_partiteness, _X_is_multipartite


class BaseMultipartiteEstimator(BaseEstimator):
    """Base class for multipartite estimators.
    """
    # _partiteness should be set to k for an estimator that accepts only
    # k-partite input (in fit(), k = len(X) = y.ndim). None implies that any
    # any k >= 2 is accepted.
    _partiteness = None

    def score(self, X, y, sample_weight=None):
        # TODO: multipartite multi-output (y.ndim = 3).
        return super().score(X, y.reshape(-1), sample_weight)

    def _more_tags(self):
        # Determines that it can receive X as a list of Xs, one for each y axis,
        # that is, a list of objects of any type in self._get_tags()["X_types"]
        return dict(multipartite=True)

    def _validate_data(
        self,
        X="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        **check_params,
    ):
        for Xax in X:
            self._check_feature_names(Xax, reset=reset)

        if y is None and self._get_tags()["requires_y"]:
            raise ValueError(
                f"This {self.__class__.__name__} estimator "
                "requires y to be passed, but the target y is None."
            )

        no_val_X = isinstance(X, str) and X == "no_validation"
        no_val_y = y is None or isinstance(y, str) and y == "no_validation"

        default_check_params = {"estimator": self}
        check_params = {**default_check_params, **check_params}

        if no_val_X and no_val_y:
            raise ValueError("Validation should be done on X, y or both.")
        elif not no_val_X and no_val_y:
            # TODO: better way of deciding. We still accept nd_arrays in
            #       predict, considering them as molten multipartite X
            #       (see docs for :module:melter)
            if _X_is_multipartite(X):
                check_partiteness(X, estimator=self)
                for ax in range(len(X)):
                    X[ax] = check_array(X[ax], input_name="X", **check_params)
            else:
                X = check_array(X, input_name="X", **check_params)
            out = X
        elif no_val_X and not no_val_y:
            check_partiteness(y=y, estimator=self)
            y = _check_y(y, **check_params)
            out = y
        else:
            check_partiteness(X, y, estimator=self)
            if len(X) != y.ndim:
                raise ValueError("Incompatible X and y given. The number of "
                                 "attribute matrices (len(X)) must correspond "
                                 f"to y's number of dimensions, but {len(X)=} "
                                 f"and {y.ndim=}.")
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                if "estimator" not in check_X_params:
                    check_X_params = {**default_check_params, **check_X_params}

                for ax in range(len(X)):
                    X[ax] = check_array(X[ax], input_name="X", **check_X_params)

                if "estimator" not in check_y_params:
                    check_y_params = {**default_check_params, **check_y_params}
                y = check_array(y, input_name="y", **check_y_params)
            else:
                X[0], y = check_X_y(X[0], y, multi_output=True, **check_params)
                _y = y
                for ax in range(1, len(X)):
                    _y = np.moveaxis(_y, 0, -1)
                    X[ax], _ = check_X_y(X[ax], _y, multi_output=True,
                                         **check_params)
            out = X, y

        if not no_val_X and check_params.get("ensure_2d", True):
            self._check_n_features(X, reset=reset)

        return out

    def _check_n_features(self, X, reset):
        """Set the `n_features_in_` attribute, or check against it.
        Parameters
        ----------
        X : list of {ndarray, sparse matrix} of shapes (n_samples[i], \
        n_features[i])
            The input samples.
        reset : bool
            If True, the `n_features_in_` attribute is set to
            `sum(Xi.shape[1] for Xi in X)`.
            If False and the attribute exists, then check that it is equal to
            `X.shape[1]`. If False and the attribute does *not* exist, then
            the check is skipped.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.
        """
        if not _X_is_multipartite(X):  # Catches molten X (see ../melter.py)
            return super()._check_n_features(X, reset=reset)
        try:
            axes_n_features = tuple(_num_features(Xi) for Xi in X)
            n_features = sum(axes_n_features)
        except TypeError as e:
            if not reset and hasattr(self, "n_features_in_"):
                raise ValueError(
                    "X does not contain any features, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{self.n_features_in_} features"
                ) from e
            # If the number of features is not defined and reset=True,
            # then we skip this check
            return

        if reset:
            self.n_features_in_ = n_features
            self.axes_n_features_in_ = axes_n_features
            return

        if not hasattr(self, "n_features_in_"):
            # Skip this check if the expected number of expected input features
            # was not recorded by calling fit first. This is typically the case
            # for stateless transformers.
            return

        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )


class BaseBipartiteEstimator(BaseMultipartiteEstimator):
    _partiteness = 2


class MultipartiteTransformerMixin(TransformerMixin):
    """Mixin for multipartite transformers."""


# FIXME: make it consistent with imblearn hierarchy
class MultipartiteSamplerMixin(BaseMultipartiteEstimator, SamplerMixin):
    pass


class BaseMultipartiteSampler(MultipartiteSamplerMixin):
    """Base class for multipartite samplers.
    """
    sampling_strategy = "auto"
    _sampling_type = "clean-sampling"

    def fit_resample(self, X, y):
        # TODO: do not bypass input validation.
        return self._fit_resample(X, y)

    def _check_X_y(self, X, y, accept_sparse=None):
        if accept_sparse is None:
            accept_sparse = ["csr", "csc"]
        # y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = self._validate_data(X, y, reset=True, accept_sparse=accept_sparse)
        return X, y, None

    def _more_tags(self):
        return {"X_types": ["n_partite", "2darray", "sparse", "dataframe"]}


# =============================================================================
# FIXME: The following classes should not be used. They are just a prototype.
# =============================================================================


class BaseData(metaclass=ABCMeta):
    @abstractmethod
    def check(self):
        """Check validity of the data provided."""


class BaseInputData(BaseData, metaclass=ABCMeta):
    def __init__(self, X, y=None, check=False):
        self.X = X
        self.y = y
        self.Xy = X, y
        if check:
            self.check()

    @abstractmethod
    def __getitem__(*indices):
        ...
    
    def check(self, estimator=None):
        """Check validity of the data provided."""
        return self

    @abstractproperty
    def n_samples(self):
        """The number of samples in the dataset."""

    @abstractproperty
    def n_features(self):
        """The number of features in the dataset."""


class StandardInputData(BaseInputData):
    @property
    def n_samples(self):
        pass

    @property
    def n_features(self):
        pass


class PairwiseInputData(BaseInputData):
    def __getitem__(self, indices):
        X_ = X[np.ix_(indices, indices)]
        y_ = y and y[indices]
        return self.__class__(X_, y_)


class InputDataND(BaseInputData):
    def __getitem__(self, indices):
        X_ = [Xi[np.r_[idx]] for Xi, idx in zip(self.X, indices)]
        if self.y is None:
            y_ = None
        else:
            y_ = self.y[np.ix_(*(np.r_[idx] for idx in indices))]
        return self.__class__(X_, y_)

    @property
    def n_samples(self):
        # TODO: Use sklearn's _n_samples()
        return tuple(Xi.shape[0] for Xi in self.X)

    @property
    def n_features(self):
        return tuple(Xi.shape[1] for Xi in self.X)

    @property
    def n_dimensions(self):
        return len(self.X)
