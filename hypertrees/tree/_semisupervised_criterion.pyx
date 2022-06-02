from sklearn.tree._criterion cimport Criterion, RegressionCriterion
from sklearn.tree._criterion import MSE
from sklearn.tree._tree cimport SIZE_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset
import numpy as np
cimport numpy as cnp


cdef class SemisupervisedCriterion(Criterion):
    """Base class for semantic purposes and future maintenance.
    """


cdef class SSRegressionCriterion(SemisupervisedCriterion):
    """Base class for semantic purposes and future maintenance.
    """


# Maybe "SSEnsembleCriterion"
cdef class SSCompositeCriterion(SemisupervisedCriterion):
    """Combines results from two criteria to yield its own.
    
    One criteria will receive y in its init() and the other will receive X.
    Their calculated impurities will then be combined as the final impurity:

        sup*supervised_impurity + (1-sup)*unsupervised_impurity

    where sup is self.supervision.

    When training with an unsupervised criterion, one must provide X and y
    stacked (joined cols) as the y parameter of the estimator's fit(). E.g.:

    >>> clf = DecisionTreeRregressor(criterion=ss_criterion)
    >>> clf.fit(X=X, y=np.hstack([X, y]))
    """
    def __reduce__(self):
        return (
            SSCompositeCriterion,
            (
                self.supervision,
                None,
                self.supervised_criterion,
                self.unsupervised_criterion,
            ),
            self.__getstate__(),
        )

    def __init__(
        self,
        double supervision=.5,
        criterion=None,
        supervised_criterion=None,
        unsupervised_criterion=None,
        SIZE_t n_outputs=0,
        SIZE_t n_features=0,
        SIZE_t n_samples=0,
        *args, **kwargs,
    ):
        if not (0 <= supervision <= 1):
            # TODO: == 0 only for tests.
            raise ValueError("supervision must be in [0, 1] interval.")

        if criterion is None and (supervised_criterion is None or
                                  unsupervised_criterion is None):
            raise ValueError('If criterion is None, both supervised and unsupe'
                            'rvised criteria must be given.')
        supervised_criterion = supervised_criterion or criterion
        unsupervised_criterion = unsupervised_criterion or criterion

        if isinstance(supervised_criterion, type):
            if not n_outputs or not n_samples:
                raise ValueError('If supervised_criterion is a class, one must'
                                 ' provide both n_outputs (received '
                                 f'{n_outputs}) and n_samples ({n_samples}).')
            supervised_criterion = supervised_criterion(
                n_outputs=n_outputs,
                n_samples=n_samples,
            )
        if isinstance(unsupervised_criterion, type):
            if not n_features or not n_samples:
                raise ValueError('If unsupervised_criterion is a class, one mu'
                                 'st provide both n_features (received '
                                 f'{n_features}) and n_samples ({n_samples}).')
            unsupervised_criterion = unsupervised_criterion(
                n_outputs=n_features,
                n_samples=n_samples,
            )

        self.supervision = supervision
        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion
        self.n_outputs = self.supervised_criterion.n_outputs
        self.n_samples = self.supervised_criterion.n_samples
        self.n_features = self.unsupervised_criterion.n_outputs

    cdef int init(
            self, const DOUBLE_t[:, ::1] y,
            DOUBLE_t* sample_weight,
            double weighted_n_samples, SIZE_t* samples, SIZE_t start,
            SIZE_t end) nogil except -1:
        # y will actually be X and y concatenated.
        self.X = y[:, :self.n_features]
        self.y = y[:, self.n_features:]
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end-start
        self.weighted_n_samples = weighted_n_samples

        # TODO: SCAFFOLD
        with gil:
            print('*** SS CRIT INIT sup', self.supervision)
            print('*** SS CRIT INIT y', self.y.shape)
            print('*** SS CRIT INIT X', self.X.shape)
            print('*** SS CRIT n_features, n_outputs:',
                self.n_features,
                self.n_outputs,
            )
            print('*** SS CRIT', self.supervised_criterion.n_outputs, self.supervised_criterion.n_samples)
            print('*** SS CRIT', self.unsupervised_criterion.n_outputs, self.unsupervised_criterion.n_samples)

        self.supervised_criterion.init(
            self.y, sample_weight, weighted_n_samples, samples, start, end,
        )
        self.unsupervised_criterion.init(
            self.X, sample_weight, weighted_n_samples, samples, start, end,
        )

        # TODO: the stuff below is also calculated by the second splitter,
        # we should find a good way of calculating it only once.
        # FIXME: this is probably wrong. Criterion.impurity_improvement fails
        # when the values below are used.
        self.weighted_n_node_samples = \
            self.supervised_criterion.weighted_n_node_samples
        self.weighted_n_left = \
            self.supervised_criterion.weighted_n_left
        self.weighted_n_right = \
            self.supervised_criterion.weighted_n_right

        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criteria at pos=start."""
        if self.supervised_criterion.reset() == -1:
            return -1
        if self.unsupervised_criterion.reset() == -1:
            return -1
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criteria at pos=end."""
        if self.supervised_criterion.reverse_reset() == -1:
            return -1
        if self.unsupervised_criterion.reverse_reset() == -1:
            return -1
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left child.
        This updates the collected statistics by moving samples[pos:new_pos]
        from the right child to the left child.
        Parameters
        ----------
        new_pos : SIZE_t
            New starting index position of the samples in the right child
        """
        cdef int rc
        if self.supervised_criterion.update(new_pos) == -1:
            return -1
        if self.unsupervised_criterion.update(new_pos) == -1:
            return -1
        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        """Calculate the impurity of the node.
        Impurity of the current node, i.e. the impurity of samples[start:end].
        This is the primary function of the criterion class. The smaller the
        impurity the better.
        """
        cdef double sup = self.supervision

        return (
            sup * self.supervised_criterion.node_impurity() + \
            (1-sup) * self.unsupervised_criterion.node_impurity()
        )

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Calculate the impurity of children.
        Evaluate the impurity in children nodes, i.e. the impurity of
        samples[start:pos] + the impurity of samples[pos:end].

        Parameters
        ----------
        impurity_left : double pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : double pointer
            The memory address where the impurity of the right child should be
            stored
        """
        cdef double s_impurity_left, s_impurity_right
        cdef double u_impurity_left, u_impurity_right
        cdef double sup = self.supervision

        self.supervised_criterion.children_impurity(
            &s_impurity_left, &s_impurity_right,
        )
        self.unsupervised_criterion.children_impurity(
            &u_impurity_left, &u_impurity_right,
        )

        impurity_left[0] = sup*s_impurity_left + (1-sup)*u_impurity_left
        impurity_right[0] = sup*s_impurity_right + (1-sup)*u_impurity_right
        with gil:  # TODO: scaffold
            print('*** SS CRIT INIT imp left/right', impurity_left[0], impurity_right[0])

    cdef void node_value(self, double* dest) nogil:
        """Store the node value.
        Compute the node value of samples[start:end] and save the value into
        dest.

        Parameters
        ----------
        dest : double pointer
            The memory address where the node value should be stored.
        """
        self.supervised_criterion.node_value(dest)

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction.
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef double sup = self.supervision
        return \
            sup * self.supervised_criterion.proxy_impurity_improvement() + \
            (1-sup) * self.unsupervised_criterion.proxy_impurity_improvement()

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil:
        cdef double sup = self.supervision
        cdef double s_imp = self.supervised_criterion.impurity_improvement(
            impurity_parent, impurity_left, impurity_right)
        cdef double u_imp = self.unsupervised_criterion.impurity_improvement(
            impurity_parent, impurity_left, impurity_right)

        return sup*s_imp + (1-sup)*u_imp


cdef class SSMSE(SSCompositeCriterion):
    """Applies MSE both on supervised (X) and unsupervised (y) data.
    
    One criteria will receive y in its init() and the other will receive X.
    Their calculated impurities will then be combined as the final impurity:

        sup*supervised_impurity + (1-sup)*unsupervised_impurity

    where sup is self.supervision.
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
            criterion=MSE,
            n_outputs=n_outputs,
            n_features=n_features,
            n_samples=n_samples,
        )


# TODO
# cdef class SSCriterionWrapper(RegressionCriterionWrapper2D):
#     pass