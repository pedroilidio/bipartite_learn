from libc.math cimport log2
import numpy as np
cimport numpy as cnp

from sklearn.tree._tree cimport SIZE_t
from sklearn.tree._criterion import MSE

from ._semisupervised_criterion cimport SSCompositeCriterion


cdef class DynamicSSMSE(SSCompositeCriterion):
    """ It's SSMSE, but it changes its supervision value each run. 
    
    One criteria will receive y in its init() and the other will receive X.
    Their calculated impurities will then be combined as the final impurity:

        sup*supervised_impurity + (1-sup)*unsupervised_impurity

    where sup is self.supervision.

    Note that the splitter holding it should receive X and y concatenated
    (horizontally stacked) as its y parameter (y=np.hstack((X, y))).
    """
    def __init__(
        self,
        double supervision,
        SIZE_t n_outputs,
        SIZE_t n_features,
        SIZE_t n_samples,
        *args, **kwargs,
    ):
        super().__init__(
            supervision=supervision,
            supervised_criterion=MSE,
            unsupervised_criterion=MSE,
            n_outputs=n_outputs,
            n_features=n_features,
            n_samples=n_samples,
        )

    def update_supervision(self):
        cdef double w, W

        w = self.weighted_n_node_samples 
        W = self.weighted_n_samples

        return 1/(1 + 2 ** (10*(.5 - log2(w)/log2(W))))
        # self.weighted_n_node_samples / weighted_n_samples


cdef class HyperbolicSSCriterion(SSCompositeCriterion):
    """ It's SSMSE, but changes its supervision value each run. 
    
    One criteria will receive y in its init() and the other will receive X.
    Their calculated impurities will then be combined as the final impurity:

        sup*supervised_impurity + (1-sup)*unsupervised_impurity

    where sup is self.supervision.

    Note that the splitter holding it should receive X and y concatenated
    (horizontally stacked) as its y parameter (y=np.hstack((X, y))).
    """

    def update_supervision(self):
        """Hyperbolic function asymptotically going to 1 when depth increases.

        self.supervision increases with tree depth.

        self.original_supervision (supervision parameter from __init__) will
        represent the supervision at halfway depth (log2(W)/2).

        So, if self.original_supervision=.80, one should expect that a full
        grown tree is already 80% supervised at half its maximum depth.
        """
        cdef double w, W, beta, unsup

        w = self.weighted_n_node_samples 
        W = self.weighted_n_samples
        unsup = 1 - self.original_supervision

        beta = log2(unsup) / log2(W)

        return 1 - (unsup / w ** beta) ** 2