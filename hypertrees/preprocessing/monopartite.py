from __future__ import annotations
from typing import Callable, Literal
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.validation import check_symmetric
from imblearn.base import BaseSampler
from ..utils import check_similarity_matrix

_GammaScaleOptions = Literal["constant", "squares", "squared_errors", "size"]


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


def _scale_gamma(X, gamma: float, gamma_scale: str) -> float:
    """Multiply gamma by an X-dependent factor.
    Parameters
    ----------
    X : 2d ndarray
        Input array of kernel function.
    gamma : float
        `gamma` parameter to be passed to kernel function.
    gamma_scale : {'constant', 'squares', 'squared_errors', 'size'}
        If not 'constant', divide `gamma` by `S / y.shape[0]`, where
        `S = (y**2).sum()`, if `gamma_scale='squares'`,
        `((y-y.mean()) ** 2).sum()` if 'squared_errors' and 'y.size' if 'size'.
    Returns
    -------
    scaled gamma : float
    """
    if gamma_scale == "constant":
        return gamma
    elif gamma_scale == "squares":
        return gamma * X.shape[0] / (X**2).sum()
    elif gamma_scale == "squared_errors":
        return gamma / (X.shape[1] * X.var())
    elif gamma_scale == "size":
        return gamma / X.shape[1]
    else:
        raise ValueError(
            "Unrecognized gamma_scale: {gamma_scale!r}. Valid options are "
            "'constant', 'squares', 'squared_errors' and 'size'."
        )


class TargetKernelLinearCombiner(BaseSampler):
    """Combines provided similarity matrix X with kernel calculated over y

    X is assumed to be a precomputed kernel matrix. The target kernel will be
    calculated with `sklearn.metrics.pairwise.pairwise_kernels` and combined
    with X simply by taking `alpha*X + (1-alpha)*y_kernel`.

    The default kernel is RBF, so that it calculates the 'gaussian interaction
    profile' as described by [1].

    Valid values for metric are:
        ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf',
        'laplacian', 'sigmoid', 'cosine']

    Parameters
    ----------
    alpha : float, default=0.5
        Controls the fraction of the target information in the linear
        combination with the provided similarities. alpha=1 means no change,
        alpha=0 means no original X data will remain.
    metric : str or callable, default="rbf"
        The metric to use when calculating kernel between instances in a
        feature array. If metric is a string, it must be one of the metrics
        in `sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS`.
        If metric is "precomputed", y is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two rows from y as input and return the corresponding
        kernel value as a single number. This means that callables from
        :mod:`sklearn.metrics.pairwise` are not allowed, as they operate on
        matrices, not single samples. Use the string identifying the kernel
        instead.
    gamma : float, default=1.0
        `gamma` parameter of kernel function if metric is callable, 'chi2',
        'polynomial', 'rbf', 'laplacian' or 'sigmoid'.
    gamma_scale : {'constant', 'squares', 'squared_errors', 'size'}, \
    default='squares'
        If not 'constant', divide `gamma` by `S / y.shape[0]`, where
        `S = (y**2).sum()`, if `gamma_scale='squares'`,
        `((y-y.mean()) ** 2).sum()` if 'squared_errors' and 'y.size' if 'size'.
    filter_params : bool, default=False
        Whether to filter invalid kernel parameters or not.
    n_jobs : int, default=None
        The number of jobs to use for the kernel computation. This works by
        breaking down the y matrix into n_jobs even slices and computing
        them in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the kernel function.
    References
    ----------
    .. [1] :doi:`"Gaussian interaction profile kernels for predicting drug–\
       target interaction" <https://doi.org/10.1093/bioinformatics/btr500>`
       van Laarhoven, Nabuurs and Marchiori, 2011.
    """
    def __init__(
        self,
        alpha: float = 0.5,
        metric: str | Callable = "rbf",
        gamma: float = 1.0,
        gamma_scale: _GammaScaleOptions = "squares",
        filter_params: bool = False,
        n_jobs: int | None = None,
        **kwds,
    ):
        self.alpha = alpha
        self.metric = metric
        self.gamma = gamma
        self.gamma_scale = gamma_scale
        self.filter_params = filter_params
        self.n_jobs = n_jobs
        self._kernel_params = kwds

    def _fit_resample(self, X, y, **fit_params):
        self.effective_gamma_ = _scale_gamma(y, self.gamma, self.gamma_scale)
        y_kernel = pairwise_kernels(
            y,
            metric=self.metric,
            gamma=self.effective_gamma_,
            filter_params=self.filter_params,
            n_jobs=self.n_jobs,
            **self._kernel_params,
        )
        return self.alpha * X + (1-self.alpha) * y_kernel, y


class TargetKernelDiffuser(BaseSampler):
    """Calculates kernel on y and performs non-linear kernel diffusion.

    DOI: https://doi.org/10.1016/j.aca.2016.01.014
    Hao _et al._, 2016.

    X is assumed to be a precomputed kernel matrix. The target kernel will be
    calculated with `sklearn.metrics.pairwise.pairwise_kernels` and combined
    with X by a kernel diffusion procedure [1].

    The default kernel is RBF, so that it calculates the 'gaussian interaction
    profile' as described by [2].

    Valid values for metric are:
        ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf',
        'laplacian', 'sigmoid', 'cosine']

    Parameters
    ----------
    n_iter : int, default=2
        Number of diffusion iterations.
    n_neighbors : int, default=4
        n_neighbors parameter passed to kneighbors_graph for local similarity
        calculation.
    metric : str or callable, default="rbf"
        The metric to use when calculating kernel between instances in a
        feature array. If metric is a string, it must be one of the metrics
        in `sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS`.
        If metric is "precomputed", y is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two rows from y as input and return the corresponding
        kernel value as a single number. This means that callables from
        :mod:`sklearn.metrics.pairwise` are not allowed, as they operate on
        matrices, not single samples. Use the string identifying the kernel
        instead.
    gamma : float, default=1.0
        `gamma` parameter of kernel function if metric is callable, 'chi2',
        'polynomial', 'rbf', 'laplacian' or 'sigmoid'.
    gamma_scale : {'constant', 'squares', 'squared_errors', 'size'}, \
    default='squares'
        If not 'constant', divide `gamma` by `S / y.shape[0]`, where
        `S = (y**2).sum()`, if `gamma_scale='squares'`,
        `((y-y.mean()) ** 2).sum()` if 'squared_errors' and 'y.size' if 'size'.
    filter_params : bool, default=False
        Whether to filter invalid kernel parameters or not.
    n_jobs : int, default=None
        The number of jobs to use for the kernel computation. This works by
        breaking down the y matrix into n_jobs even slices and computing
        them in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the kernel function.
    References
    ----------
    .. [1] :doi:`"Improved prediction of drug-target interactions using \
       regularized least squares integrating with kernel fusion technique" \
        <https://doi.org/10.1016/j.aca.2016.01.014>`
        Ming Hao, Yanli Wang and Bryant, Stephen H, 2016.
    .. [2] :doi:`"Gaussian interaction profile kernels for predicting drug–\
       target interaction" <https://doi.org/10.1093/bioinformatics/btr500>`
       van Laarhoven, Nabuurs and Marchiori, 2011.
    """
    def __init__(
        self,
        n_iter: int = 2,
        n_neighbors: int = 4,
        metric: str | Callable = "rbf",
        gamma: float = 1.0,
        gamma_scale: _GammaScaleOptions = "squares",
        filter_params: bool = False,
        n_jobs: int | None = None,
        **kwds,
    ):
        self.n_iter = n_iter
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.gamma = gamma
        self.gamma_scale = gamma_scale
        self.filter_params = filter_params
        self.n_jobs = n_jobs
        self._kernel_params = kwds

    def _fit_resample(self, X, y):
        y_kernel = pairwise_kernels(
            y,
            metric=self.metric,
            gamma=_scale_gamma(y, self.gamma, self.gamma_scale),
            filter_params=self.filter_params,
            n_jobs=self.n_jobs,
            **self._kernel_params,
        )
        S = X.copy()  # Similarity matrix
        S_net = y_kernel.copy()  # Connections-based similarity

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

        # NOTE: Not in the original paper.
        S_final /= S_final.max(axis=1, keepdims=True)  # Normalize to [0, 1]

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
        S_local = kneighbors_graph(
            2 - S,  # Convert similarity to distance
            n_neighbors=n_neighbors,
            metric="precomputed",
            mode="distance",
        )  # sparse matrix
        # Convert non-zero entries back to similarities
        S_local.data = 2 - S_local.data
        # Normalize rows
        S_local /= S_local.sum(axis=1)  # Dimension is kept by default
        return np.array(S_local)


class SymmetryEnforcer(BaseSampler):
    """Make matrix symmetric by averaging it with its transpose.
    """
    def _fit_resample(self, X, y):
        return check_symmetric(X), y

class PositiveSemidefiniteEnforcer(BaseEstimator, TransformerMixin):
    """Modify main diagonal to enforce positive semidefiniteness.

    Adds to the main diagonal the absolute value of the minimum negative
    eigen-value, returning a positive-semidefinite version of X.

    Parameters
    ----------
    tol : float, default=1e-5
        To avoid small negative values due to numerical precision,
        tol is also added to the diagonal.
    """
    def __init__(self, tol=1e-5):
        self.tol = tol

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return enforce_positive_semidefiniteness(X, self.tol)


class SimilarityDistanceSwitcher(BaseEstimator, TransformerMixin):
    """Transforms x into (1 - x)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return 1 - check_similarity_matrix(X, check_symmetry=False)

    def inverse_transform(self, X):
        return self.transform(X)
