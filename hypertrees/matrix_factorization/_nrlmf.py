"""
This is a module to be used as a reference for building other modules
"""
# TODO: lazy knn
from typing import Sequence
from random import random
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state
from sklearn.neighbors import KNeighborsRegressor
from ..base import RegressorMixinND


DEF_KNN_PARAMS = dict(
    n_neighbors=5,
    metric="precomputed",
    weights=lambda x: -x,  # Use similarities instead of distances.
    algorithm="brute",
)


def check_n_partite_params(*params, n=2):
    new_params = []

    for p in params:
        if isinstance(p, Sequence) and not isinstance(p, str):
            if len(p) != n:
                raise ValueError
            new_params.append(p)
        else:
            new_params.append((p,) * n)

    return new_params


class NRLMF(RegressorMixinND, BaseEstimator, BaseSampler):
    """Neighborhood Regularized Logistic Matrix Factorization.

    Optionally, one can define beta normalization.

    [1] Yong Liu, Min Wu, Chunyan Miao, Peilin Zhao, Xiao-Li Li, "Neighborhood
    Regularized Logistic Matrix Factorization for Drug-target Interaction
    Prediction"
    """

    def __init__(
        self,
        weight_positives=5,
        k_neighbors=5,
        n_components=10,
        learning_rate=1.0,
        alpha=0.1,
        latent_prior_std=0.625,
        eta1=7,
        eta2=3,
        max_iter=100,
        change_positives=False,
        resample_X=False,
        random_state=None,
        knn_args=None,
    ):
        # importance level for positive observations
        self.weight_positives = weight_positives
        self.learning_rate = learning_rate
        self.latent_prior_std = latent_prior_std
        self.alpha = alpha
        self.eta1 = eta1
        self.eta2 = eta2
        self.max_iter = max_iter
        self.change_positives = change_positives
        self.random_state = random_state
        self.resample_X = resample_X
        self.knn_args = knn_args

    # def _more_tags(self):
    #     return dict(pairwise=True)

    @staticmethod
    def _logistic_output(U, V):
        """Compute probabilities of interaction.
        
        Calculate interaction probabilities (y predictions) based on the given
        U and V latent feature vectors.
        """
        P = np.exp(self.U @ self.V.T)
        return P / (P + 1)

    def predict(X, y=None):
        # FIXME: they use all neighbors in the paper.
        U = self.knn_rows_.predict(X[0])
        V = self.knn_cols_.predict(X[1])
        yt = self._logistic_output(U, V)

        # If an eta provided, perform Beta rescoring (Ban _et al._, 2019)
        # FIXME: it should be done over the original y, and without weight_positives
        # TODO: review
        if self.eta1 != 0:
            yt = self.beta_rescore(X, yt)

        return yt

    # FIXME: should actually be:
    # def _fit_resample(self, X, y):
    def fit_resample(self, X, y):
        self.fit(X, y)

        yt = self._logistic_output(self.U, self.V)
        Xt = [self.U, self.V]

        if not self.change_positives:
            # Transform y only where it was zero.
            yt[y == 1] = 1.

        # If an eta provided, perform Beta rescoring (Ban _et al._, 2019)
        # TODO: review
        if self.eta1 != 0:
            yt = self.beta_rescore(Xt, yt)

        if self.transform_X:
            return Xt, yt
        else:
            return X, yt

    def fit(self, X, y):
        # TODO: Use sklearn.metrics.pairwise.check_pairwise_array or similar.
        # FIXME: improve input checking.
        random_state = check_random_state(self.random_state)
        alpha, latent_prior_std, knn_params = check_n_partite_params(
            self.alpha, self.latent_prior_std, self.knn_params,
        )

        # Initialize U and V latent vectors.
        sqrt_n_comp = np.sqrt(self.n_components)
        self.U = random_state.normal(size=(X[0].shape[0], self.n_components))
        self.V = random_state.normal(size=(X[1].shape[0], self.n_components))
        self.U /= sqrt_n_comp
        self.V /= sqrt_n_comp

        knn_params[0] = DEF_KNN_PARAMS | (knn_params[0] or {})
        knn_params[1] = DEF_KNN_PARAMS | (knn_params[1] or {})
        self.knn_rows_ = KNeighborsRegressor(**knn_params[0])
        self.knn_cols_ = KNeighborsRegressor(**knn_params[1])
        self.knn_rows_.fit(X[0], self.U)
        self.knn_cols_.fit(X[1], self.V)

        self.n_features_in_ = X[0].shape[1] + X[1].shape[1]

        # To be used in gradient calculation.
        self.y_scaled_ = 1 + (self.weight_positives-1) * y

        # Build regularized K Nearest Neighbors similarity matrices.
        L_rows = self._laplacian_matrix(
            alpha=alpha[0], knn=self.knn_rows_,
            latent_prior_std=latent_prior_std[0],
        )
        L_cols = self._laplacian_matrix(
            alpha=alpha[1], knn=self.knn_cols_,
            latent_prior_std=latent_prior_std[1],
        )

        # Optimize U and V latent vectors for X[0] and X[1]
        self._AGD_optimization(
            U=self.U, V=self.V, L_rows=L_rows, L_cols=L_cols, y=y
        )

        return self

    def _laplacian_matrix(self, alpha, latent_prior_std, knn):
        """Calculate the neighborhood regularization matrix.

        We deviate from the definition in Liu _et al._'s Eq. 11 by including
        the constants to be used in gradient calculation.  The return value
        thus actually corresponds to 
        
        \\lambda \\mathbf{I} + \\alpha \mathbf{L},

        according to the paper's definition, in order to facilitate usage on
        Equations 13. `lambda_` will be \\lambda_d or \\lambda_t. `const` will
        be \\alpha or \\beta.

        Note: knn must already be fitted.
        """
        S_knn = knn.kneighbors_graph(mode="distance")
        DD = np.sum(S_knn, axis=0) + np.sum(S_knn, axis=1)

        return (np.diag(DD+self.latent_prior_std) - (S_knn+S_knn.T)) * alpha / 2

    def _AGD_optimization(self, U, V, L_rows, L_cols, y):
        """Find U and V values to minimize the loss function.

        AdaGrad procedure to determine latent feature vectors under the
        neighborhood regularized logistic loss.
        """
        step_sq_sum_rows = np.zeros_like(U)
        step_sq_sum_cols = np.zeros_like(V)
        last_log = self._log_likelihood(y, U, V, L_rows, L_cols)

        for i in range(self.max_iter):
            # FIXME: Is the alternation correct?
            if i%2:
                step = self._gradient_step(y, U, V, L_rows)
                step_sq_sum_rows += step ** 2
                U += self.learning_rate * step / np.sqrt(step_sq_sum_rows)
            else:
                step = self._gradient_step(y.T, V, U, L_cols)
                step_sq_sum_cols += step ** 2
                V += self.learning_rate * step / np.sqrt(step_sq_sum_cols)

            curr_log = self._log_likelihood(y, U, V, L_rows, L_cols)
            delta_log = abs(1 - curr_log/last_log)

            if delta_log < self.tol:
                break

            last_log = curr_log

    def _gradient_step(self, y, W, otherW, L):
        """Calculate a step against the loss function gradient.

        Implements equations 13 from Liu _et al._[1]. Note that we redefine L
        in `self._laplacian_matrix()` to include alpha and lambda constants.

        Parameters
        ----------
        y: array of 0s and 1s
            The original binary interaction matrix.
        W: array of floats
            Can be U or V. If W = U, otherW is V, and vice-versa.
        otherW: array of floats
            The other set of latent vectors.
        L: array of floats
            Neighborhood regularized matrix (See Eq. 11 of [1]), but with
            lambda and alpha constants included, as returned by
            `self._laplacian_matrix()`.
        """
        # Current predictions for probability of interaction.
        P = np.exp(W @ otherW.T)
        P = P / (P+1)

        # NOTE: the following is the negative gradient, so that it already
        #       climbs down the loss function and must be directly added to W.
        return (self.weight_positives*y - self.y_scaled_*P) @ otherW - L @ W

    def _log_likelihood(self, y, U, V, L_rows, L_cols):
        A = U @ V.T
        B = A * y
        A =  np.log(np.exp(A) + 1) * self.y_scaled_

        return (
            np.sum(B) - np.sum(A)
            - self.lambda_d * np.sum(U**2)
            - self.lambda_t * np.sum(V**2)
            - self.alpha * np.sum(np.diag(U.T @ L_rows) @ U)
            - self.beta * np.sum(np.diag(V.T @ L_cols) @ V)
        ) / 2
