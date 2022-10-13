# TODO: docs
# TODO: KNN sampler to be used by NRLMF
# TODO: Beta correction kernel aggregator
# TODO: MultipartiteSamplerWrapper
from __future__ import annotations
from numbers import Number
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import kneighbors_graph
from imblearn.base import BaseSampler

from ..utils import check_multipartite_params
from ..base import BaseMultipartiteSampler, BaseMultipartiteEstimator
from .multipartite_transformers import MultipartiteTransformerWrapper


def nearest_positive_semidefinite(X):
    """Get nearest (Frobenius norm) positive semidefinite matrix from A.
    
    See Equations (2.1) and (2.2) of [1].
    Also see [https://stackoverflow.com/q/43238173/11286509].

    [1] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    A = (X + X.T)/2
    eigval, eigvec = np.linalg.eig(A)
    eigval[eigval < 0] = 0
    return (eigvec * eigval) @ eigvec.T


# nope
def npsd(X):
    val = np.linalg.eigvals(X)
    mval = val[val<0].min()
    X = X.copy()
    np.fill_diagonal(X, X.diagonal() - mval)
    return X


def make_symmetric(X):
    return (X + X.T) / 2


class KernelSupervisedTransformer(BaseSampler):
    ...


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


# TODO: kernel combination (linear or fusion) should be in a separate object
# TODO: is the name "kernel diffusion"?
class NonlinearKernelFusionSampler(BaseMultipartiteSampler):
    """Kernel Fusion with GIP as described by Hao _et al._, 2016.

    DOI: https://doi.org/10.1016/j.aca.2016.01.014

    Parameters
    ----------
    n_iter : int, default=2
        ... TODO

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
    def __init__(self, n_iter=2, alpha=0.5, length_scale=None,
                 length_scale_bounds=1.0, knn_params=None):
        self.n_iter = n_iter
        self.alpha = alpha
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.knn_params = knn_params

    def _local_fit_resample(self, X, y, **fit_params):
        # FIXME: distance, not similarity
        knn_params = dict(
            n_neighbors=4,
            metric="precomputed",
            mode="distance",
        ) | (self.knn_params or {})

        # TODO: pass list of length_scale and length_scale_bounds for each
        #       axis.
        length_scale = self.length_scale or np.sqrt(np.mean(y**2)/2)
        net_similarity = RBF(length_scale, self.length_scale_bounds)(y)
        similarity = X.copy()
        I = np.identity(like=similarity)

        # TODO: add small multiple of I (?)
        similarity = (similarity + similarity.T + I) / 2
        similarity /= similarity.sum(axis=1)

        net_similarity = (net_similarity + net_similarity.T + I) / 2
        net_similarity /= net_similarity.sum(axis=1)

        local_sim = kneighbors_graph(1/similarity, **knn_params)
        local_net_sim = kneighbors_graph(1/net_similarity, **knn_params)
        # nonzero_idx = S_knn.nonzero()
        # S_knn[nonzero_idx] = 1 / local_sim[nonzero_idx]

        for _ in range(self.n_iter):
            similarity = local_sim @ net_similarity @ local_sim.T
            net_similarity = local_net_sim @ similarity @ local_net_sim.T
            similarity = (similarity + similarity.T + I) / 2
            net_similarity = (net_similarity + net_similarity.T + I) / 2
        
        final_similarity = (similarity + net_similarity) / 2
        final_similarity = (final_similarity + final_similarity.T + I) / 2
        final_similarity /= final_similarity.sum(axis=1)

        return final_similarity



# TODO: transformer?
class KernelSymmetryEnforcerSampler(BaseSampler):
    """Make kernel matrix simmetric by averaging it with its transpose.
    """
    def _fit_resample(self, X, y):
        X_resampled = [
            (Xi + Xi.T) / 2
            for Xi in X
        ]
        return X_resampled, y


# XXX
class EnforcePositiveSemidefiniteTransformer(BaseSampler):
    """Modify main diagonal to enforce positive semidefiniteness.

    Add values to the kernels' main diagonal to make it positive semidefinite.
    """
    def fit_transform(self, X, y=None):
        return X 


class SimilarityToDistanceTransformer(BaseMultipartiteEstimator):
    """Transforms x into 1/x
    """
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
