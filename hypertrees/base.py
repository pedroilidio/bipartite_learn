import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import _check_y
from imblearn.base import BaseSampler, SamplerMixin


class BaseMultipartiteEstimator(BaseEstimator):
    """Base class for multipartite estimators.
    """
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
            if isinstance(X, (list, tuple)):  # TODO: better way of deciding.
                for ax in range(len(X)):
                    X[ax] = check_array(X[ax], input_name="X", **check_params)
            else:
                X = check_array(X, input_name="X", **check_params)
            out = X
        elif no_val_X and not no_val_y:
            y = _check_y(y, **check_params)
            out = y
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                if "estimator" not in check_X_params:
                    check_X_params = {**default_check_params, **check_X_params}

                if isinstance(X, (list, tuple)):
                    for ax in range(len(X)):
                        X[ax] = check_array(X[ax], input_name="X", **check_params)
                else:
                    X = check_array(X, input_name="X", **check_params)

                if "estimator" not in check_y_params:
                    check_y_params = {**default_check_params, **check_y_params}
                y = check_array(y, input_name="y", **check_y_params)
            else:
                raise NotImplementedError("set validate_separately=True")
                X, y = check_X_y(X, y, **check_params)
            out = X, y

        if not no_val_X and check_params.get("ensure_2d", True):
            pass  # TODO
            # self._check_n_features(X, reset=reset)

        return out


class MultipartiteRegressorMixin(RegressorMixin):
    """Mixin for multipartite regressors.
    """

    def score(self, X, y, sample_weight=None):
        # TODO: multi-output.
        return super().score(X, y.reshape(-1), sample_weight)


class MultipartiteTransformerMixin(TransformerMixin):
    """Mixin for multipartite transformers."""


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
        #y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
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
