from sklearn.tree._criterion cimport Criterion
from sklearn.tree._splitter cimport Splitter
from ._sklearn_splitter cimport (
    BestSplitter, 
#     BaseDenseSplitter, RandomSplitter,
#     BaseSparseSplitter, BestSparseSplitter, RandomSparseSplitter 
)
import numpy as np
cimport numpy as np
from sklearn.tree._tree import DOUBLE


cdef class SSCriterion(Criterion):
    """Abstract base class for semisupervised criteria."""

    cdef int init(
            self, const DOUBLE_t[:, ::1] y,
            DOUBLE_t* sample_weight,
            double weighted_n_samples, SIZE_t* samples, SIZE_t start,
            SIZE_t end) nogil except -1:

        with gil:
            print("You must use semisupervised_init() with semisupervised "
                  "Criteria, not init().")
        return -1

    cdef int semisupervised_init(
            self, const DOUBLE_t[:, ::1] X, const DOUBLE_t[:, ::1] y,
            DOUBLE_t* sample_weight,
            double weighted_n_samples, SIZE_t* samples, SIZE_t start,
            SIZE_t end) nogil except -1:
        pass


# Maybe "SSEnsembleCriterion"
cdef class SSCompositeCriterion(SSCriterion):
    """Combines results from two criteria to yield its own.
    
    One criteria will receive y in its init() and the other will receive X.
    """
    def __init__(
        self,
        Criterion supervised_criterion,
        Criterion unsupervised_criterion,
        double supervision,
    ):
        if not (0 <= supervision <= 1):
            # TODO: == 0 only for tests.
            raise ValueError("supervision must be in [0, 1] interval.")
        self.supervision = supervision
        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion
        self.n_outputs = supervised_criterion.n_outputs
        self.n_samples = supervised_criterion.n_samples

    cdef int semisupervised_init(
            self, const DOUBLE_t[:, ::1] X, const DOUBLE_t[:, ::1] y,
            DOUBLE_t* sample_weight,
            double weighted_n_samples, SIZE_t* samples, SIZE_t start,
            SIZE_t end) nogil except -1:

        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end-start
        self.weighted_n_samples = weighted_n_samples

        self.supervised_criterion.init(
            y, sample_weight, weighted_n_samples, samples, start, end,
        )
        self.unsupervised_criterion.init(  # TODO: some stuff recalculated 
            X, sample_weight, weighted_n_samples, samples, start, end,
        )

        self.weighted_n_node_samples = \
            self.supervised_criterion.weighted_n_node_samples
        self.weighted_n_left = \
            self.supervised_criterion.weighted_n_left
        self.weighted_n_right = \
            self.supervised_criterion.weighted_n_right

        self.sum_total = self.supervised_criterion.sum_total
        self.sum_left = self.supervised_criterion.sum_left
        self.sum_right = self.supervised_criterion.sum_right
        ### Only RegressionCriteria have sq_sum_total.
        # self.sq_sum_total = self.supervised_criterion.sq_sum_total

        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        if self.supervised_criterion.reset() == -1:
            return -1
        if self.unsupervised_criterion.reset() == -1:
            return -1
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
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

        return self.supervised_criterion.proxy_impurity_improvement() * sup + \
               self.unsupervised_criterion.proxy_impurity_improvement()*(1-sup)


cdef class SSBestSplitter(BestSplitter):
    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state):
        # Parent __cinit__ is automatically called.
        self.sscriterion = criterion

    cdef int init(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight) except -1:
        # Call parent init
        BestSplitter.init(self, X, y, sample_weight)

        #if getattr(X, "dtype", None) != DOUBLE_t or not X.flags.contiguous:
        self.X_targets = np.ascontiguousarray(self.X, dtype=DOUBLE)

        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1:
        self.start = start
        self.end = end

        self.sscriterion.semisupervised_init(
            self.X_targets,  # NOTE: The only difference.
            self.y,
            self.sample_weight,
            self.weighted_n_samples,
            self.samples,
            start,
            end,
        )

        weighted_n_node_samples[0] = self.sscriterion.weighted_n_node_samples
        return 0