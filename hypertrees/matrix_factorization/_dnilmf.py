import numpy as np
from sklearn.utils import check_random_state
from sklearn.neighbors import KNeighborsRegressor
from ..base import BaseMultipartiteSampler, MultipartiteRegressorMixin
from ..utils import check_multipartite_params, lazy_knn_weights_min_one

__all__ = ["DNILMF"]


DEF_KNN_PARAMS = dict(
    n_neighbors=2,
    metric="precomputed",
    algorithm="brute",
)


class DNILMF(
    BaseMultipartiteSampler,
    MultipartiteRegressorMixin,
):
    # TODO: fix documentation, KNN.
    """Dual-Network Integrated Logistic Matrix Factorization

    Note: the kernel fusion pre-processing procedure described by [1] is
    implemented as

    :module:preprocessing.monopartite_transformers.TargetKernelDiffuser

    and can be applyied together with DNILMF in a pipeline.

    Parameters
    ----------
    positive_importance : int, default=6
        The multiplier factor to apply to positive (known) interactions.
        Each positive interaction (y == 1) will weight `positive_importance`
        times more than a negative, as if we have `positive_importance`
        times more occurences of positive labels in the dataset than we
        actually have, that is, as if each positive instance was repeated
        (oversampled) `positive_importance` times. Called c in the original
        paper [1].

    n_components_rows : int, default=90
        Number of components of X[0] latent vectors, the number of columns of
        U.

    n_components_cols : int or "same", default="same"
        Number of components of X[0] latent vectors, the number of columns of
        U. If "same", it takes the same value of `n_components_rows`

    learning_rate : float or sequence of floats, default=1.0
        Multiplicative factor for each gradient step.

    alpha : float, default=0.4
        Constant that multiplies the y matrix when computing the loss function.
        The greater it is, the more supervised is the algorithm. Must satisfy
        `alpha + beta + gamma == 1`.

    beta : float, default=0.3
        Constant that multiplies the row similarity matrix when computing the
        loss function. Thus, it controls the importance given by the
        algorithm to the rows's unsupervised information. Must satisfy
        `alpha + beta + gamma == 1`.

    gamma : float, default=0.3
        Constant that multiplies the column similarity matrix when computing
        the loss function. Thus, it controls the importance given by the
        algorithm to the column's unsupervised information. Must satisfy
        `alpha + beta + gamma == 1`.

    lambda_rows : float, default=0.625
        Corresponds to the inverse of the assumed prior variance of U. It
        multiplies the regularization term of U.

    lambda_cols : float or "same", default="same"
        Corresponds to the inverse of the assumed prior variance of V. It
        multiplies the regularization term of V. If "same", it takes the same
        value of `lambda_rows`.

    knn_params : dict or sequence of dicts or None
        Parameters to be passed on to a `KNeighborsRegressor`, that
        will be used to get unsupervised neighborhood information. `None`
        assumes default values bellow, that can be updated by passing a
        `dict` instead. A two separate dictionaries, for X[0] and X[1]
        can also be provided in a sequence (usually list or tuple).

        Defaults are:
            - n_neighbors=2,
            - metric="precomputed",
            - algorithm="brute",

    max_iter : int, default=100
        Maximum number of iterations.

    tol : float, default=1e-5
        Minimum relative loss improvement to continue iteration.

    change_positives : bool, default=False
        If `False`, it keeps 1s from the original y in the transformed y.  

    resample_X : bool, default=False
        If `True`, return [U, V] as resampled X in `fit_resample`.

    verbose : bool, default=False
        Wether to display or not training status information.

    random_state : int, RandomState instance or None, default=None
        Used for initialisation of U and V. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    References
    ----------
    .. [1] :doi:`"Predicting drug-target interactions by dual-network integrated
    logistic matrix factorization" <https://doi.org/10.1038/srep40376>`
    Hao, M., Bryant, S. & Wang, Y. Sci Rep 7, 40376 (2017). 

    .. [2] :doi:`"Neighborhood Regularized Logistic Matrix Factorization for
    Drug-target Interaction Prediction" <10.1371/journal.pcbi.1004760>`
    Yong Liu, Min Wu, Chunyan Miao, Peilin Zhao, Xiao-Li Li, (2016)
    """
    # FIXME:
    # We need this for other scikit-learn stuff to not look for predict_proba()
    # however, NRLMF is primarily a Xy transformer (i.e. sampler).
    _estimator_type = "regressor"

    def __init__(
        self,
        positive_importance=6,
        n_components_rows=90,
        n_components_cols="same",
        learning_rate=1.0,
        alpha=0.4,
        beta=0.3,
        gamma=0.3,
        lambda_rows=2,
        lambda_cols="same",
        knn_params=None,
        max_iter=100,
        tol=1e-5,
        change_positives=False,
        resample_X=False,
        verbose=False,
        random_state=None,
    ):
        self.positive_importance = positive_importance
        self.learning_rate = learning_rate
        self.n_components_rows = n_components_rows
        self.n_components_cols = n_components_cols
        self.lambda_rows = lambda_rows
        self.lambda_cols = lambda_cols
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.change_positives = change_positives
        self.knn_params = knn_params
        self.max_iter = max_iter
        self.tol = tol
        self.resample_X = resample_X
        self.verbose = verbose
        self.random_state = random_state

    def _more_tags(self):
        return dict(pairwise=True)

    def _merge_similarities(self, M, L_rows, L_cols):
        return (
            self.alpha * M
            + self.beta * L_rows @ M
            + self.gamma * M @ L_cols
        )

    def _logistic_output(self, U, V, L_rows, L_cols):
        """Compute probabilities of interaction.
        
        Calculate interaction probabilities (y predictions) based on the given
        U and V latent feature vectors.

        L_rows and L_cols are kernels (similarities) among row instances and
        column instances, respectively.
        """
        UV = U @ V.T
        P = np.exp(self._merge_similarities(UV, L_rows, L_cols))
        return P / (P + 1)

    def _fit_resample(self, X, y):
        self.fit(X, y)
        yt = self._logistic_output(self.U, self.V, *X)

        if not self.change_positives:
            # Transform y only where it had zero similarity (distance == 1).
            yt[y == 1] = 1

        if self.resample_X:
            return [self.U, self.V], yt

        return X, yt

    def predict(self, X):
        U = self.knn_rows_.predict(X[0])
        V = self.knn_cols_.predict(X[1])
        yt = self._logistic_output(U, V, *X)
        return yt.reshape(-1)

    def fit(self, X, y):
        # NOTE: Fit must receive similarity matrices, contrary to NRLMF.
        # FIXME: Improve input checking.
        #        Use sklearn.metrics.pairwise.check_pairwise_array or similar.
        if (self.alpha + self.beta + self.gamma) != 1:
            raise ValueError("alpha, beta and gamma must sum to 1.")

        n_components_rows = self.n_components_rows

        if self.n_components_cols == "same":
            n_components_cols = n_components_rows
        else:
            n_components_cols = self.n_components_cols

        random_state = check_random_state(self.random_state)
        knn_params = check_multipartite_params(self.knn_params)

        # Initialize U and V latent vectors.
        self.U = random_state.normal(size=(X[0].shape[0], n_components_rows))
        self.V = random_state.normal(size=(X[1].shape[0], n_components_cols))
        self.U /= np.sqrt(n_components_rows)
        self.V /= np.sqrt(n_components_cols)

        # Initialize auxiliary KNN regressors.
        knn_params[0] = DEF_KNN_PARAMS | (knn_params[0] or {})
        knn_params[1] = DEF_KNN_PARAMS | (knn_params[1] or {})
        self.knn_rows_ = KNeighborsRegressor(**knn_params[0])
        self.knn_cols_ = KNeighborsRegressor(**knn_params[1])
        self.knn_rows_.fit(X[0], self.U)
        self.knn_cols_.fit(X[1], self.V)

        self.n_features_in_ = X[0].shape[1] + X[1].shape[1]

        # Optimize U and V latent vectors for X[0] and X[1].
        self._AGD_optimization(
            U=self.U, V=self.V, L_rows=X[0], L_cols=X[1], y=y,
        )

        return self

    def _AGD_optimization(self, U, V, L_rows, L_cols, y):
        """Find U and V values to minimize the loss function.

        AdaGrad procedure to determine latent feature vectors under the
        neighborhood regularized logistic loss.

        Each step will be divided by the sum of all previous steps, so that the
        further we travel on the error landscape, the smaller our steps get.
        This sum will actually be the square root of the sum of squared steps.

        The loss function is the log-likelihood of U an V given y, as defined
        by [1] and [2].
        """
        lambda_rows = self.lambda_rows
        lambda_cols = lambda_rows if self.lambda_cols == "same" else self.lambda_cols

        # TODO: Is this really AdaGrad?
        # See [Duchi et al., 2011](https://jmlr.org/papers/v12/duchi11a.html)
        step_sq_sum_rows = np.zeros_like(U)
        step_sq_sum_cols = np.zeros_like(V)

        # y_scaled[i, j] = positive_importance if y[i, j] == 1 else 1
        y_scaled = 1 + (self.positive_importance-1) * y

        last_loss = self._loss_function(
            y, y_scaled, U, V, L_rows, L_cols, lambda_rows, lambda_cols,
        )

        # TODO: optimize. step_rows and step_cols could be stored in the same
        #       matrix, not a new matrix every time.
        for i in range(self.max_iter):
            P = self._logistic_output(U, V, L_rows, L_cols)
            # In the paper [1] notation, yP is now cY - Q
            yP = y*self.positive_importance - y_scaled*P  
            yP = self._merge_similarities(yP, L_rows.T, L_cols.T)
            # TODO: Check if it transposing is correct

            # Update U (derived from Eq. 7 of [1], see the class's docstring).
            step_rows = yP @ V - lambda_rows * U
            step_sq_sum_rows += step_rows ** 2
            U += self.learning_rate * step_rows / np.sqrt(step_sq_sum_rows)

            P = self._logistic_output(U, V, L_rows, L_cols)
            yP = y*self.positive_importance - y_scaled*P
            yP = self._merge_similarities(yP, L_rows.T, L_cols.T)

            # Update V (derived from Eq. 8 of [1], see the class's docstring).
            step_cols = yP.T @ U - lambda_cols * V
            step_sq_sum_cols += step_cols ** 2
            V += self.learning_rate * step_cols / np.sqrt(step_sq_sum_cols)

            curr_loss = self._loss_function(
                y, y_scaled, U, V, L_rows, L_cols, lambda_rows, lambda_cols,
            )
            delta_loss = abs(1 - curr_loss/last_loss)

            if self.verbose:
                print(f"Step: {i+1} | Current loss: {curr_loss} | "
                      f"Relative change: {delta_loss}")

            if delta_loss < self.tol:
                break

            last_loss = curr_loss

    def _loss_function(
        self, y, y_scaled, U, V, L_rows, L_cols, lambda_rows, lambda_cols,
    ):
        """Return the loss, based on the log-likelihood of U an V given y.

        Implements Eq. 6 of [1], multiplied by -1 to minimize instead of
        maximize. L_rows and L_cols were originally called S_d and S_t in the
        paper, being rows' and columns' similarity matrices, respectively.
        """
        UV = self._merge_similarities(U @ V.T, L_rows, L_cols)
        return (
            np.sum(
                y_scaled * np.log(1 + np.exp(UV))
                - y * UV * self.positive_importance
            )
            + lambda_rows/2 * (U**2).sum()
            + lambda_cols/2 * (V**2).sum()
        )
