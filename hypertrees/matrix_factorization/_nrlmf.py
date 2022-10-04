import numpy as np
from sklearn.utils import check_random_state
from sklearn.neighbors import KNeighborsRegressor
from ..base import BaseMultipartiteSampler, MultipartiteRegressorMixin
from ..utils import check_multipartite_params, lazy_knn_weights_min_one

__all__ = ["NRLMF"]


DEF_KNN_PARAMS = dict(
    n_neighbors=5,
    metric="precomputed",
    weights=lazy_knn_weights_min_one,
    algorithm="brute",
)


class NRLMF(
    BaseMultipartiteSampler,
    MultipartiteRegressorMixin,
):
    """Neighborhood Regularized Logistic Matrix Factorization.

    Important note: contrary to the original paper's definition, this estimator
    must receive **distance** matrices, not similarity matrices as the X
    parameter for fit(). One can just invert similarity matrices to use this
    class, for example, utilizing

    :module:preprocessing.bipartite_samplers.SimilarityToDistanceTransformer

    in an `imblearn.pipeline.Pipeline`.

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

    n_components : int or sequence of int, default=10
        Number of components of X latent vectors, the number of columns of
        U and V.

    learning_rate : float or sequence of floats, default=1.0
        Multiplicative factor for each gradient step.

    alpha : float or sequence of floats, default=1.0
        Parameter of loss function. Individual values for each bipartite
        set can be passed in a sequence. Originally called \\alpha and
        \\beta.

    inverse_prior_var : float or sequence of floats, default=0.625
        Parameter of loss function. Corresponds to the inverse of the
        assumed prior variance of U and V. Individual values for each
        bipartite set (corresponding to U and V) can be passed in a
        sequence. Originally called \\lamda_t and \\lambda_d.

    knn_params : dict or sequence of dicts or None
        Parameters to be passed on to a `KNeighborsRegressor`, that
        will be used to get unsupervised neighborhood information. `None`
        assumes default values bellow, that can be updated by passing a
        `dict` instead. A two separate dictionaries, for X[0] and X[1]
        can also be provided in a sequence (usually list or tuple).

        Defaults are:
            - n_neighbors=5,
            - metric="precomputed",
            - weights=lazy_knn_weights_min_one,  # Reuse known instances
            - algorithm="brute",

    max_iter : int, default=100
        Maximum number of iterations.

    tol : float, default=1e-5
        Minimum loss value to stop iteration.

    change_positives : bool, default=False
        If `False`, it keeps 1s from the original y in the transformed y.  

    resample_X : bool, default=False
        If `True`, return [U, V] as resampled X in `fit_resample`.

    random_state : int, RandomState instance or None, default=None
        Used for initialisation of U and V. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.
    """
    # FIXME:
    # We need this for other scikit-learn stuff to not look for predict_proba()
    # however, NRLMF is primarily a Xy transformer (i.e. sampler).
    _estimator_type = "regressor"  

    def __init__(
        self,
        positive_importance=5,
        n_components=10,
        learning_rate=1.0,
        alpha=0.1,
        inverse_prior_var=0.625,
        knn_params=None,
        max_iter=100,
        tol=1e-5,
        change_positives=False,
        resample_X=False,
        random_state=None,
    ):
        self.positive_importance = positive_importance
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.inverse_prior_var = inverse_prior_var
        self.alpha = alpha
        self.max_iter = max_iter
        self.change_positives = change_positives
        self.random_state = random_state
        self.resample_X = resample_X
        self.knn_params = knn_params
        self.tol = tol

    # TODO: see how KNN deals with kernels to set pairwise tag.
    # def _more_tags(self):
    #     return dict(pairwise=True)

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

        if not self.change_positives:
            # Transform y only where it had zero similarity (distance == 1).
            yt[y == 1] = 1

        if self.resample_X:
            return [self.U, self.V], yt

        return X, yt

    def fit_predict(self, X, y):
        # NOTE: should be equal to calling fit() and then predict(), but that's
        #       only true with weights=lazy_knn_weights_min_one in knn_params.
        _, yt = self.fit_resample(X, y)
        return yt.reshape(-1)

    def predict(self, X):
        U = self.knn_rows_.predict(X[0])
        V = self.knn_cols_.predict(X[1])
        yt = self._logistic_output(U, V)
        return yt.reshape(-1)

    def fit(self, X, y):
        # NOTE: Fit must receive distance matrices, not similarity matrices,
        #       contrary to the paper's description of the problem [1].
        # FIXME: Improve input checking.
        #        Use sklearn.metrics.pairwise.check_pairwise_array or similar.
        random_state = check_random_state(self.random_state)
        alpha, inverse_prior_var, knn_params = check_multipartite_params(
            self.alpha, self.inverse_prior_var, self.knn_params,
        )

        # Initialize U and V latent vectors.
        denominator = np.sqrt(self.n_components)
        self.U = random_state.normal(size=(X[0].shape[0], self.n_components))
        self.V = random_state.normal(size=(X[1].shape[0], self.n_components))
        self.U /= denominator
        self.V /= denominator

        # Initialize auxiliary KNN regressors.
        knn_params[0] = DEF_KNN_PARAMS | (knn_params[0] or {})
        knn_params[1] = DEF_KNN_PARAMS | (knn_params[1] or {})
        self.knn_rows_ = KNeighborsRegressor(**knn_params[0])
        self.knn_cols_ = KNeighborsRegressor(**knn_params[1])
        self.knn_rows_.fit(X[0], self.U)
        self.knn_cols_.fit(X[1], self.V)

        self.n_features_in_ = X[0].shape[1] + X[1].shape[1]

        # To be used in gradient calculation.
        self.y_scaled_ = 1 + (self.positive_importance-1) * y

        # Build regularized K Nearest Neighbors similarity matrices.
        L_rows = self._laplacian_matrix(
            alpha=alpha[0], knn=self.knn_rows_,
            inverse_prior_var=inverse_prior_var[0],
        )
        L_cols = self._laplacian_matrix(
            alpha=alpha[1], knn=self.knn_cols_,
            inverse_prior_var=inverse_prior_var[1],
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
        
        \\lambda \\mathbf{I} + \\alpha \mathbf{L},

        according to the paper's definition, in order to facilitate usage on
        Equations 13. `inverse_prior_var` will be \\lambda_d or \\lambda_t.
        `alpha` will be \\alpha or \\beta.

        Note: knn must already be fitted by `self.fit()`.
        """
        S_knn = knn.kneighbors_graph(mode="distance")
        nonzero_idx = S_knn.nonzero()

        # We need similarities but fit() needs to provide a distance matrix to
        # the KNeighborsRegressor at `knn`. So S_knn will initially have
        # distances that we must to convert to similarities. Thus, invert it.
        S_knn[nonzero_idx] = alpha / S_knn[nonzero_idx]

        S_knn = S_knn.toarray()

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
        last_loss = self._log_likelihood(y, U, V, L_rows, L_cols)

        # TODO: optimize. step_rows and step_cols should be stored in the same
        #       matrix, not a new matrix every time.
        for i in range(self.max_iter):
            # Update U.
            step_rows = self._gradient_step(y, self.y_scaled_, U, V, L_rows)
            step_sq_sum_rows += step_rows ** 2
            U += self.learning_rate * step_rows / np.sqrt(step_sq_sum_rows)

            # Update V.
            step_cols = self._gradient_step(y.T, self.y_scaled_.T, V, U, L_cols)
            step_sq_sum_cols += step_cols ** 2
            V += self.learning_rate * step_cols / np.sqrt(step_sq_sum_cols)

            # Calculate loss.
            curr_loss = self._log_likelihood(y, U, V, L_rows, L_cols)
            delta_loss = abs(1 - curr_loss/last_loss)

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

    def _log_likelihood(self, y, U, V, L_rows, L_cols):
        """Return the loss function, i.e. the log-likelihood of U an V given y.

        Implements Eq. 12 of [1]. Notice that we defined L to include the
        \\alpha and \\lambda constants (see docs for `self._laplacian_matrix`).

        We also used that np.trace(A.T @ B) == np.sum(A * B).
        """
        UV = U @ V.T

        return (
            np.sum(
                self.y_scaled_ * np.log(1 + np.exp(UV))
                - y * UV * self.positive_importance
            )
            + np.sum(U * (L_rows @ U)) / 2
            + np.sum(V * (L_cols @ V)) / 2
        )
