# TODO: docs
from __future__ import annotations
from numbers import Number
import numpy as np
from sklearn.gaussian_process.kernels import RBF

from ..utils import check_multipartite_params
from ..base import BaseMultipartiteSampler, BaseMultipartiteEstimator
from .multipartite_transformers import MultipartiteTransformerWrapper


class DTHybridSampler(BaseMultipartiteSampler):
    def __init__(self, lamb=.5, alpha=.5):
        self.lamb = lamb
        self.alpha = alpha

    def _fit_resample(self, X, y):
        S_row_net = y @ y.T  # Rows network similarity
        k_row = np.diag(S_row_net).copy()  # Row nodes degree
        # If the row makes no interactions, set the similarity to 0
        # (bellow we will divide the similarity by k_row)
        # TODO: raise warning.
        k_row[k_row == 0] = np.inf

        # k_row_total[i, j] = k_row[i] + k_row[j]
        k_row_total = k_row[:, np.newaxis] * k_row

        # k_row_weighted[i, j] = lambda**k_row[i] + k_row[j]**(1-lambda)
        k_row_weighted = (
            k_row[:, np.newaxis]**self.lamb * k_row**(1-self.lamb)
        )

        k_col = y.sum(axis=0)  # Column nodes degree
        # If the column makes no interactions, set the similarity to 0
        k_col[k_col == 0] = np.inf

        # Row similarity based on column chemical similarity
        S_row_from_column = y @ X[1] @ y.T / k_row_total

        S = self.alpha * X[0] + (1-self.alpha) * S_row_from_column
        Gamma = S / k_row_weighted

        W = Gamma * ((y / k_col) @ y.T)

        return X, W @ y
    

class GaussianInteractionProfileSampler(BaseMultipartiteSampler):
    """GIP kernel as described by van Laarhoven _et al._, 2011.

    DOI: https://doi.org/10.1093/bioinformatics/btr500

    Parameters
    ----------
    alpha : float, default=0.5
        Controls the fraction of the GIP in the linear combination with
        the provided similarities. alpha=1 means no change, alpha=0 means no
        original X data will remain.

    length_scale : float, nd.array[float], None, default=None
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each
        dimension of l defines the length-scale of the respective feature
        dimension. If None, row/column averages are used.

    length_scale_bounds : float, default=(1e-5, 1e5)
        The lower and upper bound on `length_scale`. If set to “fixed”,
        `length_scale` cannot be changed during hyperparameter tuning.
    """
    def __init__(self, alpha=0.5, length_scale=None, length_scale_bounds=1.0):
        self.alpha = alpha
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    def _fit_resample(self, X, y, **fit_params):
        # TODO: pass list of length_scale and length_scale_bounds for each
        #       axis.
        l_rows = self.length_scale or np.sqrt(np.mean(y**2, axis=1)/2)
        l_cols = self.length_scale or np.sqrt(np.mean(y**2, axis=0)/2)

        net_similarity_rows = RBF(l_rows, self.length_scale_bounds)(y)
        net_similarity_cols = RBF(l_cols, self.length_scale_bounds)(y.T)

        new_X = [
            self.alpha*X[0] + (1-self.alpha)*net_similarity_rows,
            self.alpha*X[1] + (1-self.alpha)*net_similarity_cols,
        ]
        return new_X, y


class KernelSymmetryEnforcerSampler(BaseMultipartiteSampler):
    def __init__(self, identity_bias=0.1):
        """Make kernel matrix simmetric by averaging it with its transpose.

        Optionally a constant value can be added to the kernels' main diagonals
        to favor their positive definiteness [Van Laarhoven, 2011].

        Parameters
        ----------
        identity_bias : float | Sequence[float], default=0.
            Value to be added to kernels main diagonal.
        """
        self.identity_bias = identity_bias

    def _fit_resample(self, X, y):
        self.identity_bias_ = check_multipartite_params(self.identity_bias,
                                                        k=len(X))

        X_resampled = [
            (Xi + Xi.T) / 2 + bias * np.identity(Xi.shape[0])
            for Xi, bias in zip(X, self.identity_bias_)
        ]

        return X_resampled, y


class SimilarityToDistanceTransformer(BaseMultipartiteEstimator):
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Xt = []
        for Xi in X:
            Xti = Xi.copy()
            Xti[Xti == 0 | np.isnan(Xti)] = self.epsilon
            Xt.append(1 / Xti)
        return Xt


# TODO: define multipartite transformer ABC
class SubtractFromMaxTransformer(BaseMultipartiteEstimator):
    """Subtract similarities from fixed value (usually 1).

    Many estimators make use kernels or nearest neighbors aproaches that depend
    on distance values. Since scikit-learn does not yet provide a
    "FarthestNeighbors" or similar objects, one can use this class to subtract
    a fixed scalar value (usually 1) from the similarities in order to make the
    X matrices represent distances instead.

    Parameters
    ----------
    max_value : Number | Sequence[Number] | None, default=1
        The number to subtract similarities from. Usually the maximum
        acceptable similarity value. Different values for each partite set can
        be provided as a sequence.
    """
    def __init__(
        self,
        max_value : Number | None = 1.,
    ):
        self.max_value = max_value
    
    def fit(self, X, y=None):
        self.max_value_ = check_multipartite_params(self.max_value,
                                                    ndim=len(X))
        return self
    
    def transform(self, X, y=None):
        Xt = []
        for i, (mv, Xi) in enumerate(zip(self.max_value_, X)):
            Xti = mv - Xi
            if (Xti < 0).any():
                raise ValueError(
                    f"A value higher than {mv} was found in X[{i}], yielding"
                    "negative distance values."
                )
            Xt.append(Xti)

        return Xt