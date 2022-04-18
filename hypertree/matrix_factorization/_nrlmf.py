"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from ..base import RegressorMixinND


class NRLMF(RegressorMixinND, BaseEstimator):
    """Neighborhood Regularized Logistic Matrix Factorization.

    See [Liu _et al._, 2016](https://doi.org/10.1371/journal.pcbi.1004760).
    """
    def _more_tags(self):
        return dict(pairwise=True)

    def __init__(
            self,
            cfix=5,
            K1=5,
            K2=5,
            num_factors=10,
            theta=1.0,
            lambda_d=0.625,
            lambda_t=0.625,
            alpha=0.1,
            beta=0.1,
            max_iter=100,
            change_positives=False,
    ):
        self.cfix = cfix  # importance level for positive observations
        self.K1 = K1
        self.K2 = K2
        self.num_factors = num_factors
        self.theta = theta
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.change_positives = change_positives

    def fit_resample(self, X, y, W=1, seed=None):
        # FIXME: we are bypassing input checking.
        return self._fit_resample(X, y, W)

    def _fit_resample(self, X, y, W=1, seed=None):
        self.n_features_ = X[0].shape[1] + X[1].shape[1]
        intMat = y
        drugMat, targetMat = X

        self.num_drugs, self.num_targets = intMat.shape
        self.ones = np.ones((self.num_drugs, self.num_targets))
        self.intMat = self.cfix*intMat*W
        self.intMat1 = (self.cfix-1)*intMat*W + self.ones
        x, y = np.where(self.intMat > 0)
        self.train_drugs, self.train_targets = set(x.tolist()), set(y.tolist())
        self._construct_neighborhood(drugMat, targetMat)
        self._AGD_optimization(seed)

        new_y = 1 / (1 + 1/(np.exp(self.U @ self.V.T)))

        if not self.change_positives:
            # Change input only where it was zero.
            mask = self.intMat.astype(bool)
            new_y[mask] = self.intMat[mask]

        return X, new_y

    def _AGD_optimization(self, seed=None):
        if seed is None:
            self.U = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_targets, self.num_factors))
        else:
            prng = np.random.RandomState(seed)
            self.U = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_targets, self.num_factors))
        dg_sum = np.zeros((self.num_drugs, self.U.shape[1]))
        tg_sum = np.zeros((self.num_targets, self.V.shape[1]))
        last_log = self._log_likelihood()
        for t in range(self.max_iter):
            dg = self._deriv(True)
            dg_sum += np.square(dg)
            vec_step_size = self.theta / np.sqrt(dg_sum)
            self.U += vec_step_size * dg
            tg = self._deriv(False)
            tg_sum += np.square(tg)
            vec_step_size = self.theta / np.sqrt(tg_sum)
            self.V += vec_step_size * tg
            curr_log = self._log_likelihood()
            delta_log = (curr_log-last_log)/abs(last_log)
            if abs(delta_log) < 1e-5:
                break
            last_log = curr_log

    def _deriv(self, drug):
        if drug:
            vec_deriv = np.dot(self.intMat, self.V)
        else:
            vec_deriv = np.dot(self.intMat.T, self.U)
        A = np.dot(self.U, self.V.T)
        A = np.exp(A)
        A /= (A + self.ones)
        A = self.intMat1 * A
        if drug:
            vec_deriv -= np.dot(A, self.V)
            vec_deriv -= self.lambda_d*self.U+self.alpha*np.dot(self.DL, self.U)
        else:
            vec_deriv -= np.dot(A.T, self.U)
            vec_deriv -= self.lambda_t*self.V+self.beta*np.dot(self.TL, self.V)
        return vec_deriv

    def _log_likelihood(self):
        loglik = 0
        A = np.dot(self.U, self.V.T)
        B = A * self.intMat
        loglik += np.sum(B)
        A = np.exp(A)
        A += self.ones
        A = np.log(A)
        A = self.intMat1 * A
        loglik -= np.sum(A)
        loglik -= 0.5 * self.lambda_d * np.sum(np.square(self.U))+0.5 * self.lambda_t * np.sum(np.square(self.V))
        loglik -= 0.5 * self.alpha * np.sum(np.diag((np.dot(self.U.T, self.DL)).dot(self.U)))
        loglik -= 0.5 * self.beta * np.sum(np.diag((np.dot(self.V.T, self.TL)).dot(self.V)))
        return loglik

    def _construct_neighborhood(self, drugMat, targetMat):
        self.dsMat = drugMat - np.diag(np.diag(drugMat))
        self.tsMat = targetMat - np.diag(np.diag(targetMat))
        if self.K1 > 0:
            S1 = self._get_nearest_neighbors(self.dsMat, self.K1)
            self.DL = self._laplacian_matrix(S1)
            S2 = self._get_nearest_neighbors(self.tsMat, self.K1)
            self.TL = self._laplacian_matrix(S2)
        else:
            self.DL = self._laplacian_matrix(self.dsMat)
            self.TL = self._laplacian_matrix(self.tsMat)

    def _laplacian_matrix(self, S):
        x = np.sum(S, axis=0)
        y = np.sum(S, axis=1)
        L = 0.5*(np.diag(x+y) - (S+S.T))  # neighborhood regularization matrix
        return L

    def _get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(S[i, :])[::-1][:min(size, n)]
            X[i, ii] = S[i, ii]
        return X
