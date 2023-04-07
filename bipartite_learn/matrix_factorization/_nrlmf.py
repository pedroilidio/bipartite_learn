import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils import check_random_state
from sklearn.neighbors import KNeighborsRegressor
from ..base import BaseMultipartiteSampler
from ..utils import check_similarity_matrix, _X_is_multipartite

__all__ = ["NRLMF"]


class NRLMF(
    BaseMultipartiteSampler,
    RegressorMixin,
):
    """Neighborhood Regularized Logistic Matrix Factorization.

    [1] Yong Liu, Min Wu, Chunyan Miao, Peilin Zhao, Xiao-Li Li, "Neighborhood
    Regularized Logistic Matrix Factorization for Drug-target Interaction
    Prediction" DOI: 10.1371/journal.pcbi.1004760

    Parameters
    ----------
    positive_importance : int, default=5
        The multiplier factor to apply to positive (known) interactions.
        Each positive interaction (y == 1) will weight `positive_importance`
        times more than a negative, as if we have `positive_importance`
        times more occurences of positive labels in the dataset than we
        actually have, that is, as if each positive instance was repeated
        (oversampled) `positive_importance` times. Called c in the original
        paper [1].

    n_components_rows : int, default=10
        Number of components of X[0] latent vectors, the number of columns of
        U.

    n_components_cols : int or "same", default="same"
        Number of components of X[0] latent vectors, the number of columns of
        U. If "same", it takes the same value of `n_components_rows`

    alpha_rows : float, default=1.0
        Constant that multiplies the local similarity matrix of row instances,
        weighting their neighborhood information when calculating the loss.

    alpha_cols : float or "same", default="same"
        Constant that multiplies the local similarity matrix of column
        instances, weighting their neighborhood information when calculating
        the loss.  Originally called :math:`\\beta` by [1].  If "same", it
        takes the same value of `alpha_rows`.

    lambda_rows : float, default=0.625
        Corresponds to the inverse of the assumed prior variance of U. It
        multiplies the regularization term of U.

    lambda_cols : float or "same", default="same"
        Corresponds to the inverse of the assumed prior variance of V. It
        multiplies the regularization term of V. If "same", it takes the same
        value of `lambda_rows`.

    n_neighbors : int, default=5
        Number of nearest neighbors to consider when predicting new samples and
        building the local similarity (laplacian) matrices.

    learning_rate : float, default=1.0
        Multiplicative factor for each gradient step.

    max_iter : int, default=100
        Maximum number of iterations.

    tol : float, default=1e-5
        Minimum relative loss improvement to continue iteration.

    keep_positives : bool, default=False
        If `True`, it keeps 1s from the original y in the transformed y.
        Note that it does not apply when calling only predict(), so that
        fit_predict() will no longer yield the same result as fit().predict().

    resample_X : bool, default=False
        If `True`, return [U, V] as resampled X in `fit_resample`.

    verbose : bool, default=False
        Wether to display or not training status information.

    random_state : int, RandomState instance or None, default=None
        Used for initialisation of U and V. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.
    """
    # NOTE: We need the next line for other scikit-learn stuff to not look for
    #       predict_proba(). However, NRLMF also implements imblearn's Sampler
    #       interface, to reconstruct y (interaction matrix) in a pipeline.
    _estimator_type = "regressor"
    _partiteness = 2

    def __init__(
        self,
        positive_importance=5,
        n_components_rows=10,
        n_components_cols="same",
        alpha_rows=0.1,
        alpha_cols="same",
        lambda_rows=0.625,
        lambda_cols="same",
        n_neighbors=5,
        learning_rate=1.0,
        max_iter=100,
        tol=1e-5,
        keep_positives=False,
        resample_X=False,
        verbose=False,
        random_state=None,
    ):
        self.positive_importance = positive_importance
        self.n_components_rows = n_components_rows
        self.n_components_cols = n_components_cols
        self.lambda_rows = lambda_rows
        self.lambda_cols = lambda_cols
        self.alpha_rows = alpha_rows
        self.alpha_cols = alpha_cols
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.keep_positives = keep_positives
        self.n_neighbors = n_neighbors
        self.tol = tol
        self.resample_X = resample_X
        self.verbose = verbose
        self.random_state = random_state

    def _more_tags(self):
        return dict(pairwise=True)

    @staticmethod
    def _logistic_output(U, V):
        """Compute probabilities of interaction.

        Calculate interaction probabilities (y predictions) based on the given
        U and V latent feature vectors.
        """
        P = np.exp(U @ V.T)
        return P / (P + 1)

    def _fit_resample(self, X, y):
        self.fit(X, y)
        yt = self._logistic_output(self.U, self.V)

        if self.keep_positives:
            # Transform y only where it was < 1.
            # The reason is that we usually assume that 1-valued labels are
            # verified interactions, while 0 represents unknown interactions.
            yt[y == 1] = 1

        if self.resample_X:
            return [self.U, self.V], yt

        return X, yt

    def fit_predict(self, X, y):
        _, yt = self.fit_resample(X, y)
        return yt.reshape(-1)

    def predict(self, X):
        if not _X_is_multipartite(X):
            raise ValueError(
                f"{type(self).__name__} only accepts bipartite input."
            )
        U = self.knn_rows_.predict(1-X[0])
        V = self.knn_cols_.predict(1-X[1])
        yt = self._logistic_output(U, V)
        return yt.reshape(-1)

    def fit(self, X, y):
        X, y = self._validate_data(X, y)

        # Fit must receive [0, 1]-bounded similarity matrices.
        for ax in range(len(X)):
            X[ax] = check_similarity_matrix(X[ax], estimator=self)

        lambda_rows = self.lambda_rows
        alpha_rows = self.alpha_rows
        n_components_rows = self.n_components_rows

        alpha_cols = \
            alpha_rows if self.alpha_cols == "same" else self.alpha_cols
        lambda_cols = \
            lambda_rows if self.lambda_cols == "same" else self.lambda_cols

        if self.n_components_cols == "same":
            n_components_cols = n_components_rows
        else:
            n_components_cols = self.n_components_cols

        random_state = check_random_state(self.random_state)

        # Initialize U and V latent vectors.
        self.U = random_state.normal(size=(X[0].shape[0], n_components_rows))
        self.V = random_state.normal(size=(X[1].shape[0], n_components_cols))
        self.U /= np.sqrt(n_components_rows)
        self.V /= np.sqrt(n_components_cols)

        # Initialize auxiliary KNN regressors.
        knn_params = dict(
            n_neighbors=self.n_neighbors,
            metric="precomputed",
            weights="distance",
            algorithm="brute",  # auto sets it to brute when metric=precomputed
        )
        self.knn_rows_ = KNeighborsRegressor(**knn_params)
        self.knn_cols_ = KNeighborsRegressor(**knn_params)
        self.knn_rows_.fit(1-X[0], self.U)  # Similarity to distance conversion
        self.knn_cols_.fit(1-X[1], self.V)

        self.n_features_in_ = X[0].shape[1] + X[1].shape[1]

        # Build regularized K Nearest Neighbors similarity matrices.
        L_rows = self._laplacian_matrix(
            alpha=alpha_rows, knn=self.knn_rows_,
            inverse_prior_var=lambda_rows,
        )
        L_cols = self._laplacian_matrix(
            alpha=alpha_cols, knn=self.knn_cols_,
            inverse_prior_var=lambda_cols,
        )

        # Optimize U and V latent vectors for X[0] and X[1].
        self._AGD_optimization(
            U=self.U, V=self.V, L_rows=L_rows, L_cols=L_cols, y=y,
        )

        return self

    def _laplacian_matrix(self, alpha, inverse_prior_var, knn):
        """Calculate the neighborhood regularization matrix.

        We deviate from the definition in Liu _et al._'s Eq. 11 by including
        the constants to be used in gradient calculation.  The return value
        thus actually corresponds to

        \\lambda \\mathbf{I} + \\alpha \\mathbf{L},

        according to the paper's definition, in order to facilitate usage on
        Equations 13. `inverse_prior_var` will be \\lambda_d or \\lambda_t.
        `alpha` will be \\alpha or \\beta.

        Note: knn must already be fitted by `self.fit()`.
        """
        S_knn = knn.kneighbors_graph(mode="distance")
        nonzero_idx = S_knn.nonzero()

        # We take 1-S to convert distances back to similarities, since knn
        # needed to be trained with distances (1-X).
        S_knn[nonzero_idx] = alpha * (1-S_knn[nonzero_idx])

        S_knn = S_knn.toarray()  # Sparse slows matrix multiplication.

        DD = np.sum(S_knn, axis=0) + np.sum(S_knn, axis=1)
        DD += inverse_prior_var

        return np.diag(DD) - (S_knn + S_knn.T)

    def _AGD_optimization(self, U, V, L_rows, L_cols, y):
        """Find U and V values to minimize the loss function.

        AdaGrad procedure to determine latent feature vectors under the
        neighborhood regularized logistic loss.

        Each step will be divided by the sum of all previous steps, so that the
        further we travel on the error landscape, the smaller our steps get.
        This sum will actually be the square root of the sum of squared steps.
        """
        # TODO: Is this really AdaGrad?
        # See [Duchi et al., 2011](https://jmlr.org/papers/v12/duchi11a.html)
        step_sq_sum_rows = np.zeros_like(U)
        step_sq_sum_cols = np.zeros_like(V)

        # TODO: eliminate y_scaled to lower memory consumption?
        # y_scaled[i, j] = positive_importance if y[i, j] == 1 else 1
        y_scaled = 1 + (self.positive_importance-1) * y

        last_loss = self._loss_function(y, y_scaled, U, V, L_rows, L_cols)

        for i in range(self.max_iter):
            # Update U.
            step_rows = self._gradient_step(y, y_scaled, U, V, L_rows)
            step_sq_sum_rows += step_rows ** 2
            U += self.learning_rate * step_rows / np.sqrt(step_sq_sum_rows)

            # Update V.
            step_cols = self._gradient_step(y.T, y_scaled.T, V, U, L_cols)
            step_sq_sum_cols += step_cols ** 2
            V += self.learning_rate * step_cols / np.sqrt(step_sq_sum_cols)

            # Calculate loss.
            curr_loss = self._loss_function(y, y_scaled, U, V, L_rows, L_cols)
            delta_loss = abs(1 - curr_loss/last_loss)

            if self.verbose:
                print(f"Step: {i+1} | Current loss: {curr_loss} | "
                      f"Relative change: {delta_loss}")

            if delta_loss < self.tol:
                break

            last_loss = curr_loss

    def _gradient_step(self, y, y_scaled, W, otherW, L):
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
        P = self._logistic_output(W, otherW)

        # NOTE: the following is the negative gradient, so that it already
        #       climbs down the loss function and must be added to,
        #       not subtracted from W.
        return (y*self.positive_importance - y_scaled*P) @ otherW - L @ W

    def _loss_function(self, y, y_scaled, U, V, L_rows, L_cols):
        """Return the loss, based on the log-likelihood of U an V given y.

        Implements Eq. 12 of [1]. Notice that we defined L to include the
        \\alpha and \\lambda constants (see docs for `self._laplacian_matrix`).

        We also used that np.trace(A.T @ B) == np.sum(A * B).
        """
        UV = U @ V.T
        return (
            np.sum(
                y_scaled * np.log(1 + np.exp(UV))
                - y * UV * self.positive_importance
            )
            + np.sum(U * (L_rows @ U)) / 2
            + np.sum(V * (L_cols @ V)) / 2
        )
