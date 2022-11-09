from __future__ import annotations
from typing import Callable
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.gaussian_process.kernels import RBF


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


def enforce_positive_semidefiniteness(X, tol=1e-5):
    """Enforce positive-semidefiniteness of kernel matrix.

    Modifies only the main diagonal of X.

    Adds to the main diagonal the absolute value of the minimum negative
    eigen-value, returning a positive-semidefinite version of X.

    Parameters
    ----------
    X : square two-dimensional ndarray
        Kernel matrix to transform
    tol : float, default=1e-5
        To avoid small negative values due to numerical precision,
        tol is also added to the diagonal 

    Returns
    -------
    ndarray with same shape as X
        Symmetric positive-semidefinite version of X.
    """
    X = (X + X.T) / 2
    assert (X >= 0).all(), "Kernel matrix must be >= 0"
    eigvals = np.linalg.eigvals(X)
    min_negative_eigval = eigvals[eigvals < 0].min()
    np.fill_diagonal(X, X.diagonal() - min_negative_eigval + tol)
    return X


# TODO: just provide the tranformer with a parameter gamma="scale", like SVR
class NormalizedRBF(RBF):
    """Calculates the Gaussian Interaction profile kernel.

    As described by van Laarhoven _et al._, 2011.
    DOI: https://doi.org/10.1093/bioinformatics/btr500
    """
    def __init__(self, *args, length_scale=1.0, **kwargs):
        self.original_length_scale = length_scale
        super().__init__(*args, length_scale=length_scale, **kwargs)

    def __call__(self, X, Y=None, eval_gradient=False):
        # FIXME already fitted error
        # TODO check axis=0
        self.length_scale = (
            self.original_length_scale * np.mean(X**2, axis=0))**.5 / 2
        return super().__call__(X, Y=None, eval_gradient=False)


class TargetKernelLinearCombiner(BaseEstimator, TransformerMixin):
    """Combines provided similarity matrix X with kernel calculated over y

    The combination is simply `alpha*X + (1-alpha)*y_kernel`.

    Parameters
    ----------
    alpha : float, default=0.5
        Controls the fraction of the target information in the linear
        combination with the provided similarities. alpha=1 means no change,
        alpha=0 means no original X data will remain.
    kernel : Callable, default=None
        Kernel function to calculate over y. NormalizedRBF is used if None is
        provided.
    kernel_args : dict, default=None 
        Arguments to kernel function.
    """
    def __init__(
        self,
        alpha: float = 0.5,
        kernel: Callable = None,
        kernel_args: dict = None,
    ):
        self.alpha = alpha
        self.kernel = kernel
        self.kernel_args = kernel_args

    def fit(self, X, y, **fit_params):
        self.y_fit_ = y
        return self

    def transform(self, X):
        return self.fit_transform(X, self.y_fit_)

    def fit_transform(self, X, y, **fit_params):
        kernel_args = self.kernel_args or {}
        # TODO: just provide the tranformer with a parameter
        #       gamma="scale" instead, like SVR and SVC do
        kernel = self.kernel or NormalizedRBF()

        self.y_fit_ = y
        S_net = kernel(y, **kernel_args)

        return self.alpha*X + (1-self.alpha)*S_net


class TargetKernelDiffuser(BaseEstimator, TransformerMixin):
    """Nonlinear kernel diffusion as described by Hao _et al._, 2016.

    Calculates a kernel over y and performs a kernel diffusion against
    precomputed X kernel.

    DOI: https://doi.org/10.1016/j.aca.2016.01.014

    Parameters
    ----------
    n_iter : int, default=2
        Number of diffusion iterations.
    n_neighbors : int, default=4
        n_neighbors parameter passed to kneighbors_graph for local similarity
        calculation.
    kernel : Callable, default=None
        Kernel function to calculate over y. NormalizedRBF is used if None is
        provided.
    kernel_args : dict, default=None 
        Arguments to kernel function.
    """
    def __init__(
        self,
        n_iter: int = 2,
        n_neighbors: int = 4,
        kernel: Callable = None,
        kernel_args: dict = None,
    ):
        self.n_iter = n_iter
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.kernel_args = kernel_args

    def fit(self, X, y, **fit_params):
        self.y_fit_ = y
        return self

    def transform(self, X):
        return self.fit_transform(X, self.y_fit_)

    def fit_transform(self, X, y, **fit_params):
        kernel_args = self.kernel_args or {}
        # TODO: just provide the tranformer with a parameter
        #       gamma="scale" instead, like SVR and SVC do
        kernel = self.kernel or NormalizedRBF()

        self.y_fit_ = y
        S = X.copy()
        S_net = kernel(y, **kernel_args)

        self._normalize_rows(S)
        self._normalize_rows(S_net)
        self._symmetrize(S)
        self._symmetrize(S_net)

        S_local = self._local_graph(S, self.n_neighbors)
        S_net_local = self._local_graph(S_net, self.n_neighbors)

        for _ in range(self.n_iter):
            S = S_local @ S_net @ S_local.T
            S_net = S_net_local @ S @ S_net_local.T
            self._symmetrize(S)
            self._symmetrize(S_net)

        S_final = (S + S_net) / 2

        self._normalize_rows(S_final)
        self._symmetrize(S_final)

        return S_final, y

    @staticmethod
    def _symmetrize(S):
        """Symmetrize and add 1 to diagonal to encourage positive-definiteness.
        """
        S += S.T
        S /= 2
        np.fill_diagonal(S, S.diagonal() + 1)

    @staticmethod
    def _normalize_rows(S):
        """Divide each value by its row's sum, so that each row sums to 1."""
        S /= S.sum(axis=1, keepdims=True)

    @staticmethod
    def _local_graph(S, n_neighbors):
        # TODO: see SpectralClustering for inspiration.
        # Using 1/S to convert from similarity to distance
        S_local = kneighbors_graph(
            1/S,
            n_neighbors=n_neighbors,
            metric="precomputed",
            mode="distance",
        )
        np.reciprocal(S_local.data, out=S_local.data)
        S_local /= S_local.sum(axis=1)  # Dimension is kept by default
        return S_local


class KernelSymmetryEnforcer(BaseEstimator, TransformerMixin):
    """Make kernel matrix simmetric by averaging it with its transpose.
    """
    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return (X + X.T) / 2


class PositiveSemidefiniteEnforcer(BaseEstimator, TransformerMixin):
    """Modify main diagonal to enforce positive semidefiniteness.

    Adds to the main diagonal the absolute value of the minimum negative
    eigen-value, returning a positive-semidefinite version of X.

    Parameters
    ----------
    tol : float, default=1e-5
        To avoid small negative values due to numerical precision,
        tol is also added to the diagonal 
    """
    def __init__(self, tol=1e-5):
        self.tol = tol

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return enforce_positive_semidefiniteness(X, self.tol)


class ReciprocalTransformer(BaseEstimator, TransformerMixin):
    """Transforms x into 1/x
    """
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Xt = X.copy()
        # FIXME: deal with nan
        Xt[Xt == 0 | np.isnan(Xt)] = self.epsilon
        Xt = 1 / Xt
        return Xt