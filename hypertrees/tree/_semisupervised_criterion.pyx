# cython: boundscheck=True
import warnings
from sklearn.tree._splitter cimport Splitter
from sklearn.tree._criterion cimport Criterion, RegressionCriterion
from sklearn.tree._criterion import MSE
from sklearn.tree._tree cimport SIZE_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset
import numpy as np
cimport numpy as cnp
from sklearn.utils._param_validation import (
    validate_params,
    Interval,
)
from ._nd_criterion cimport (
    RegressionCriterionWrapper2D,
    MSE_Wrapper2D,
)
from ._nd_criterion import InvalidAxisError


cdef class SemisupervisedCriterion(Criterion):
    """Base class for semantic purposes and future maintenance.
    """


cdef class SSRegressionCriterion(SemisupervisedCriterion):
    """Base class for semantic purposes and future maintenance.
    """


# TODO: Maybe "SSEnsembleCriterion"
cdef class SSCompositeCriterion(SemisupervisedCriterion):
    """Combines results from two criteria to yield its own.
    
    One criteria will receive y in its init() and the other will receive X.
    Their calculated impurities will then be combined as the final impurity:

        sup*supervised_impurity + (1-sup)*unsupervised_impurity

    where sup is self.supervision.

    When training with an unsupervised criterion, one must provide X and y
    stacked (joined cols) as the y parameter of the estimator's fit(). E.g.:

    >>> splitter = sklearn.tree._splitter.BestSplitter(*)
    >>> splitter.init(X=X, y=np.hstack([X, y]), *)
    >>> splitter.node_reset(*)

    Which will call

    >>> splitter.criterion.init(y=np.hstack([X, y], *)
    """
    def __reduce__(self):
        return (
            type(self),
            (
                self.supervised_criterion,
                self.unsupervised_criterion,
                self.supervision,
                self.update_supervision,
            ),
            self.__getstate__(),
        )

    def __getstate__(self):
        return {}

    def __cinit__(
        self,
        Criterion supervised_criterion,
        Criterion unsupervised_criterion,
        double supervision,
        object update_supervision=None,  # callable
    ):
        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion
        self.n_outputs = self.supervised_criterion.n_outputs
        self.n_samples = self.supervised_criterion.n_samples
        self.n_features = self.unsupervised_criterion.n_outputs

        self.supervision = supervision
        self._curr_supervision = supervision

        self.update_supervision = update_supervision
        self._supervision_is_dynamic = self.update_supervision is not None

    cdef void unpack_y(self, const DOUBLE_t[:, ::1] y) nogil:
        """Set self.X and self.y from input y.

        The semi-supervised tree class will actually provide np.hstack([X, y])
        as the y parameter to self.init(). We need to unstack it.
        """
        self.X = y[:, :self.n_features]
        self.y = y[:, self.n_features:]

    cdef int init(self, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:

        self.unpack_y(y)  # FIXME
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        
        # FIXME: HACK: allows for dynamically changing n_outputs
        self.supervised_criterion.n_outputs = self.n_outputs
        self.unsupervised_criterion.n_outputs = self.n_features

        self.supervised_criterion.init(
            self.y, sample_weight, weighted_n_samples, samples, start, end,
        )
        self.unsupervised_criterion.init(
            self.X, sample_weight, weighted_n_samples, samples, start, end,
        )

        # TODO: weighted_n_left/right is calculated by both splitters,
        # we should find a good way of calculating it only once.
        self.weighted_n_node_samples = \
            self.supervised_criterion.weighted_n_node_samples

        if self._supervision_is_dynamic:
            with gil:
                self.supervision = self.update_supervision(
                    weighted_n_samples=self.weighted_n_samples,
                    weighted_n_node_samples=self.weighted_n_node_samples,
                    criterion=self,
                )

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
        # TODO: no unsupervised data needed?
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

        return (
            sup * self.supervised_criterion.proxy_impurity_improvement()
                / self.n_outputs  # self.supervised_criterion.n_outputs
            + (1-sup) * self.unsupervised_criterion.proxy_impurity_improvement()
                / self.n_features  # self.unsupervised_criterion.n_outputs
        )

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil:
        cdef double sup = self.supervision
        cdef double s_imp = self.supervised_criterion.impurity_improvement(
            impurity_parent, impurity_left, impurity_right)
        cdef double u_imp = self.unsupervised_criterion.impurity_improvement(
            impurity_parent, impurity_left, impurity_right)
        
        # NOTE: both s_imp and u_imp impurities would actually be equal, unless
        # the two criteria calculates them in different ways.

        return sup*s_imp + (1-sup)*u_imp


# FIXME
cdef class SingleFeatureSSCompositeCriterion(SSCompositeCriterion):
    def __init__(
        self,
        double supervision,
        criterion=None,
        supervised_criterion=None,
        unsupervised_criterion=None,
        SIZE_t n_outputs=0,
        SIZE_t n_features=1,
        SIZE_t n_samples=0,
        *args, **kwargs,
    ):
        if n_features != 1:
            warnings.warn(
                f"Provided n_features={n_features}, it will be changed to 1."
                "since a single column of X will be used each time."
            )
                
        super().__init__(
            supervision=supervision,
            criterion=criterion,
            supervised_criterion=supervised_criterion,
            unsupervised_criterion=unsupervised_criterion,
            n_outputs=n_outputs,
            n_samples=n_samples,
            n_features=1,
            *args, **kwargs,
        )
        if self.unsupervised_criterion.n_outputs != 1:
            raise ValueError(
                "Unsupervised criterion must have a single output."
            )

    def set_feature(self, SIZE_t new_feature):
        self.current_feature = new_feature
        self.X = self.full_X[:, new_feature:new_feature+1]

        self.unsupervised_criterion.init(
            y=self.X,
            sample_weight=self.sample_weight,
            weighted_n_samples=self.weighted_n_samples,
            samples=self.samples,
            start=self.start,
            end=self.end,
        )
        # TODO: no need to calculate y impurity again.
        self.current_node_impurity = self.node_impurity()

    # FIXME: Unpredictable errors can arise.
    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil:
        """Since X changes with .set_feature(), we need to recalculate node
        impurity every time.
        """
        return SSCompositeCriterion.impurity_improvement(
            self, self.current_node_impurity, impurity_left, impurity_right)
            # self, self.node_impurity(), impurity_left, impurity_right)

    cdef void unpack_y(self, const DOUBLE_t[:, ::1] y) nogil:
        self.full_X = y[:, :-self.n_outputs]
        self.y = y[:, -self.n_outputs:]
        with gil:
            self.set_feature(self.current_feature)


# =============================================================================
# Bipartite Semi-supervised Criterion Wrapper
# =============================================================================


cdef class BipartiteSemisupervisedCriterion(CriterionWrapper2D):
    def __cinit__(
        self,
        Criterion unsupervised_criterion_rows,
        Criterion unsupervised_criterion_cols,
        CriterionWrapper2D supervised_bipartite_criterion,
        double supervision_rows,
        object supervision_cols="same",
        object update_supervision=None,
    ):
        self.supervision_rows = supervision_rows
        self.supervision_cols = supervision_cols

        self._curr_supervision_rows = supervision_rows
        if supervision_cols == "same":
            self._curr_supervision_cols = supervision_rows
        else:
            self._curr_supervision_cols = supervision_cols

        
    # TODO: improve validation
    def __init__(
        self,
        Criterion unsupervised_criterion_rows,
        Criterion unsupervised_criterion_cols,
        CriterionWrapper2D supervised_bipartite_criterion,
        double supervision_rows,
        object supervision_cols="same",  # double or "same"
        object update_supervision=None,  # callable
    ):
        self._supervision_is_dynamic = update_supervision is not None

        self.unsupervised_criterion_rows = unsupervised_criterion_rows
        self.unsupervised_criterion_cols = unsupervised_criterion_cols

        self.supervised_bipartite_criterion = supervised_bipartite_criterion
        self.supervised_criterion_rows = (
            supervised_bipartite_criterion.criterion_rows
        )
        self.supervised_criterion_cols = (
            supervised_bipartite_criterion.criterion_cols
        )

        self.update_supervision = update_supervision

    cdef int init(
            self,
            const DOUBLE_t[:, ::1] X_rows,
            const DOUBLE_t[:, ::1] X_cols,
            const DOUBLE_t[:, ::1] y,
            const DOUBLE_t[:, ::1] y_transposed,
            DOUBLE_t* row_sample_weight,
            DOUBLE_t* col_sample_weight,
            double weighted_n_rows,
            double weighted_n_cols,
            SIZE_t* row_samples,
            SIZE_t* col_samples,
            SIZE_t[2] start,
            SIZE_t[2] end,
        ) nogil except -1:

        # Initialize fields
        self.X_rows = X_rows
        self.X_cols = X_cols
        self.y = y
        self.y_transposed = y_transposed
        self.row_sample_weight = row_sample_weight
        self.col_sample_weight = col_sample_weight
        self.weighted_n_rows = weighted_n_rows
        self.weighted_n_cols = weighted_n_cols
        self.weighted_n_samples = weighted_n_rows * weighted_n_cols
        self.row_samples = row_samples
        self.col_samples = col_samples

        self.start[0] = start[0]
        self.start[1] = start[1]
        self.end[0] = end[0]
        self.end[1] = end[1]

        self.unsupervised_criterion_rows.init(
            y=self.X_rows,
            sample_weight=self.row_sample_weight,
            weighted_n_samples=self.weighted_n_rows,
            samples=self.row_samples,
            start=self.start[0],
            end=self.end[0],
        )
        self.unsupervised_criterion_cols.init(
            y=self.X_cols,
            sample_weight=self.col_sample_weight,
            weighted_n_samples=self.weighted_n_cols,
            samples=self.col_samples,
            start=self.start[1],
            end=self.end[1],
        )
        self.supervised_bipartite_criterion.init(
            self.X_rows,
            self.X_cols,
            self.y,
            self.y_transposed,
            self.row_sample_weight,
            self.col_sample_weight,
            self.weighted_n_rows,
            self.weighted_n_cols,
            self.row_samples,
            self.col_samples,
            self.start,
            self.end,
        )

        self.n_row_features = self.unsupervised_criterion_rows.n_outputs
        self.n_col_features = self.unsupervised_criterion_cols.n_outputs

        # Update supervision amount
        if self._supervision_is_dynamic:
            with gil:
                self._curr_supervision_rows = self.update_supervision(
                    weighted_n_samples=self.weighted_n_rows,
                    weighted_n_node_samples=self.weighted_n_node_rows,
                    current_supervision=self._curr_supervision_rows,
                    original_supervision=self.supervision_rows,
                )
                self._curr_supervision_cols = self.update_supervision(
                    weighted_n_samples=self.weighted_n_cols,
                    weighted_n_node_samples=self.weighted_n_node_cols,
                    current_supervision=self._curr_supervision_cols,
                    original_supervision=self.supervision_cols,
                )

        self.weighted_n_node_rows = (
            self.supervised_bipartite_criterion.weighted_n_node_rows
        )
        self.weighted_n_node_cols = (
            self.supervised_bipartite_criterion.weighted_n_node_cols
        )

        # Will be used by the Splitter2D to set the Tree object
        self.weighted_n_node_samples = (
            self.supervised_bipartite_criterion.weighted_n_node_samples
        )

    cdef void node_value(self, double* dest) nogil:
        self.supervised_bipartite_criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        cdef double s_imp
        cdef double u_imp
        cdef double u_imp_rows
        cdef double u_imp_cols
        cdef double sup_rows
        cdef double sup_cols
        cdef double wu_rows
        cdef double wu_cols

        sup_rows = self._curr_supervision_rows
        sup_cols = self._curr_supervision_cols

        s_imp = self.supervised_bipartite_criterion.node_impurity()
        s_imp *= (sup_rows + sup_cols) / 2

        u_imp_rows = self.unsupervised_criterion_rows.node_impurity()
        u_imp_cols = self.unsupervised_criterion_cols.node_impurity()

        wu_rows = self.n_row_features * (1-sup_rows)
        wu_cols = self.n_col_features * (1-sup_cols)
        u_imp = (
            (wu_rows * u_imp_rows + wu_cols * u_imp_cols)
            / (self.n_row_features + self.n_col_features)
        )

        return u_imp + s_imp

    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    ) nogil:
        cdef double other_u_imp
        cdef double sup, other_sup
        cdef double n_feat, other_n_feat, n_total_feat
        cdef double u_imp_left, u_imp_right
        cdef double wu, other_wu, ws
        cdef SIZE_t pos

        if axis == 0:
            other_u_imp = self.unsupervised_criterion_cols.node_impurity()

            # There is no guarantee that the splitters are using the
            # unsupervised criteria, it is a valid option to use it only for
            # choosing an axis, calculating the unsupervised impurity only
            # here. Therefore, we must ensure the unsupervised criterion is in
            # the right position.
            pos = self.supervised_criterion_rows.pos
            self.unsupervised_criterion_rows.update(pos)

            self.unsupervised_criterion_rows.children_impurity(
                &u_imp_left, &u_imp_right,
            )
            sup = self._curr_supervision_rows
            other_sup = self._curr_supervision_cols
            n_feat = self.n_row_features
            other_n_feat = self.n_col_features

        elif axis == 1:
            other_u_imp = self.unsupervised_criterion_rows.node_impurity()

            # See the previous comment.
            pos = self.supervised_criterion_cols.pos
            self.unsupervised_criterion_cols.update(pos)

            self.unsupervised_criterion_cols.children_impurity(
                &u_imp_left, &u_imp_right,
            )
            sup = self._curr_supervision_cols
            other_sup = self._curr_supervision_rows
            n_feat = self.n_col_features
            other_n_feat = self.n_row_features

        else:
            with gil:
                raise InvalidAxisError

        wu = n_feat * (1-sup)
        other_wu = other_n_feat * (1-other_sup)
        n_total_feat = n_feat + other_n_feat

        u_imp_left = (
            (wu * u_imp_left + other_wu * other_u_imp) / n_total_feat
        )
        u_imp_right = (
            (wu * u_imp_right + other_wu * other_u_imp) / n_total_feat
        )

        ws = (sup + other_sup) / 2
        self.supervised_bipartite_criterion.children_impurity(
            impurity_left, impurity_right, axis,
        )

        impurity_left[0] =  u_imp_left + ws * impurity_left[0]
        impurity_right[0] =  u_imp_right + ws * impurity_right[0]

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right,
        SIZE_t axis,
    ) nogil:
        return self.supervised_bipartite_criterion.impurity_improvement(
            impurity_parent, impurity_left, impurity_right, axis,
        )
