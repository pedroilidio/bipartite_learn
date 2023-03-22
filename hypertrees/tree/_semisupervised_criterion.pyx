# cython: boundscheck=False
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX

import numpy as np
cimport numpy as cnp

import warnings
from sklearn.tree._splitter cimport Splitter
from sklearn.tree._criterion cimport Criterion, RegressionCriterion

from sklearn.tree._tree cimport SIZE_t
from sklearn.utils._param_validation import (
    validate_params,
    Interval,
)
from ._nd_criterion import InvalidAxisError
from ._axis_criterion cimport AxisCriterion


cdef DOUBLE_t NAN = np.nan


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
        object update_supervision=None,
    ):
        self._cached_pos = SIZE_MAX
        self._root_supervised_impurity = 0.0
        self._root_unsupervised_impurity = 0.0
        self._curr_supervision = NAN

    def __init__(
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

    cdef int init(
        self, const DOUBLE_t[:, ::1] y,
        DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        SIZE_t[:] sample_indices,
        SIZE_t start,
        SIZE_t end,
    ) nogil except -1:
        self._cached_pos = SIZE_MAX  # Important to reset cached values.

        self.unpack_y(y)  # FIXME
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        
        self.supervised_criterion.init(
            self.y, sample_weight, weighted_n_samples, sample_indices, start, end,
        )
        self.unsupervised_criterion.init(
            self.X, sample_weight, weighted_n_samples, sample_indices, start, end,
        )

        # TODO: weighted_n_left/right is calculated by both splitters,
        # we should find a good way of calculating it only once.
        self.weighted_n_node_samples = \
            self.supervised_criterion.weighted_n_node_samples
        # self.weighted_n_left = self.supervised_criterion.weighted_n_left
        # self.weighted_n_right = self.supervised_criterion.weighted_n_right

        if self._supervision_is_dynamic:
            with gil:
                self._curr_supervision = self.update_supervision(
                    weighted_n_samples=self.weighted_n_samples,
                    weighted_n_node_samples=self.weighted_n_node_samples,
                    current_supervision=self._curr_supervision,
                    original_supervision=self.supervision,
                )
        if self._root_supervised_impurity == 0.0:
            self.set_root_impurities()
        
        return 0

    cdef void set_root_impurities(self) nogil:
        self._root_supervised_impurity = (
            self.supervised_criterion.node_impurity()
        )
        self._root_unsupervised_impurity = (
            self.unsupervised_criterion.node_impurity()
        )

    cdef int reset(self) nogil except -1:
        """Reset the criteria at pos=start."""
        self.supervised_criterion.reset()
        self.unsupervised_criterion.reset()
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criteria at pos=end."""
        self.supervised_criterion.reverse_reset()
        self.unsupervised_criterion.reverse_reset()
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left child.
        This updates the collected statistics by moving sample_indices[pos:new_pos]
        from the right child to the left child.
        Parameters
        ----------
        new_pos : SIZE_t
            New starting index position of the sample_indices in the right child
        """
        self.supervised_criterion.update(new_pos)
        self.unsupervised_criterion.update(new_pos)
        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        """Calculate the impurity of the node.
        Impurity of the current node, i.e. the impurity of sample_indices[start:end].
        This is the primary function of the criterion class. The smaller the
        impurity the better.
        """
        # FIXME: init is called before node_impurity by the TreeBuilder, where
        # root impurities are also set. We will redundantly calculate the root
        # impurity here as a consequence.
        if self._root_supervised_impurity == 0.0:
            self.set_root_impurities()
            return 1.0

        cdef double sup = self._curr_supervision

        return (
            sup * self.supervised_criterion.node_impurity()
            / self._root_supervised_impurity
            + (1.0-sup) * self.unsupervised_criterion.node_impurity()
            / self._root_unsupervised_impurity
        )

    cdef void ss_children_impurities(
        self,
        double* u_impurity_left,
        double* u_impurity_right,
        double* s_impurity_left,
        double* s_impurity_right,
    ) nogil:
        if self.pos == self._cached_pos:
            u_impurity_left[0] = self._cached_u_impurity_left 
            u_impurity_right[0] = self._cached_u_impurity_right
            s_impurity_left[0] = self._cached_s_impurity_left
            s_impurity_right[0] = self._cached_s_impurity_right
            return

        self.supervised_criterion.children_impurity(
            s_impurity_left, s_impurity_right,
        )
        self.unsupervised_criterion.children_impurity(
            u_impurity_left, u_impurity_right,
        )
        self._cached_pos = self.pos
        self._cached_u_impurity_left = u_impurity_left[0] 
        self._cached_u_impurity_right = u_impurity_right[0]
        self._cached_s_impurity_left = s_impurity_left[0]
        self._cached_s_impurity_right = s_impurity_right[0]

    cdef void children_impurity(
        self,
        double* impurity_left,
        double* impurity_right,
    ) nogil:
        """Calculate the impurity of children.
        Evaluate the impurity in children nodes, i.e. the impurity of
        sample_indices[start:pos] + the impurity of sample_indices[pos:end].

        Parameters
        ----------
        impurity_left : double pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : double pointer
            The memory address where the impurity of the right child should be
            stored
        """
        cdef:
            double u_impurity_left, u_impurity_right
            double s_impurity_left, s_impurity_right
            double sup = self._curr_supervision

        self.ss_children_impurities(
            &u_impurity_left,
            &u_impurity_right,
            &s_impurity_left,
            &s_impurity_right,
        )

        impurity_left[0] = (
            sup * s_impurity_left / self._root_supervised_impurity
            + (1.0-sup) * u_impurity_left / self._root_unsupervised_impurity
        )
        impurity_right[0] = (
            sup * s_impurity_right / self._root_supervised_impurity
            + (1.0-sup) * u_impurity_right / self._root_unsupervised_impurity
        )

    cdef void node_value(self, double* dest) nogil:
        """Store the node value.
        Compute the node value of sample_indices[start:end] and save the value into
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
        # FIXME: make MSE-specific, since it is much faster
        # FIXME: AxisCriterion.n_outputs (n_outputs) is not what we need.
        # FIXME: PairwiseCriterion.n_outputs (n_features) is not what we need.
        cdef:
            double sup = self._curr_supervision
            double s_imp_left
            double s_imp_right
            double u_imp_left
            double u_imp_right

        self.ss_children_impurities(
            &u_imp_left, &u_imp_right,
            &s_imp_left, &s_imp_right,
        )

        return (
            - sup * (
                self.supervised_criterion.weighted_n_left * s_imp_left
                + self.supervised_criterion.weighted_n_right * s_imp_right
            )
            / self.supervised_criterion.weighted_n_node_samples

            - (1.0 - sup) * (
                self.unsupervised_criterion.weighted_n_left * u_imp_left
                + self.unsupervised_criterion.weighted_n_right * u_imp_right
            )
            / self.unsupervised_criterion.weighted_n_node_samples

        )

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right,
    ) nogil:
        """Compute the improvement in impurity.

            sup * supervised_improvement + (1 - sup) * unsupervised_improvement
        """
        # FIXME: Since we cannot access s_impurity and u_impurity
        # separately, we have to discard this method's input and calculate
        # them again. To mitigate this problem, self.ss_children_impurities()
        # caches the previous values it calculated and reuses them if
        # self.pos == self._cached_pos.
        # FIXME: The FIXME above is not valid if the impurity improvement is
        # a linear combination of sup impurity and unsup impurity. It's then
        # only makes sense for Friedman criterion? Friedman already discards
        # the input by its own.
        # NOTE: self._cached_pos must be reset at Criterion.init() to
        # ensure it always corresponds to the current tree node.
        cdef:
            double u_impurity_parent
            double s_impurity_parent
            double u_impurity_left, u_impurity_right
            double s_impurity_left, s_impurity_right
            double u_improvement
            double s_improvement
            double sup = self._curr_supervision

        self.ss_children_impurities(
            &u_impurity_left,
            &u_impurity_right,
            &s_impurity_left,
            &s_impurity_right,
        )

        u_impurity_parent = self.unsupervised_criterion.node_impurity()
        s_impurity_parent = self.supervised_criterion.node_impurity()

        u_improvement = self.unsupervised_criterion.impurity_improvement(
            u_impurity_parent, u_impurity_left, u_impurity_right,
        )

        s_improvement = self.supervised_criterion.impurity_improvement(
            s_impurity_parent, s_impurity_left, s_impurity_right,
        )

        return sup * s_improvement + (1.0-sup) * u_improvement


cdef class HomogeneousCompositeSS(SSCompositeCriterion):
    """Composite SS Criterion for sup and unsup criteria of the same type.
    """
    cdef double _proxy_sup

    cdef void set_proxy_sup(self) nogil:
        # FIXME: weighted columns (outputs) are not considered
        self._proxy_sup = (
            self._curr_supervision
            * self._root_unsupervised_impurity
            * self.n_features  # self.unsupervised_criterion.n_outputs
            / (
                self._root_supervised_impurity
                * self.n_outputs  # self.supervised_criterion.n_outputs
            )
        )

    cdef int init(
        self,
        const DOUBLE_t[:, ::1] y,
        DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        SIZE_t[:] sample_indices,
        SIZE_t start,
        SIZE_t end,
    ) nogil except -1:
        SSCompositeCriterion.init(
            self,
            y,
            sample_weight,
            weighted_n_samples,
            sample_indices,
            start,
            end,
        )
        self.set_proxy_sup()

    cdef double proxy_impurity_improvement(self) nogil:
        return (
            self._proxy_sup
            * self.supervised_criterion.proxy_impurity_improvement()
            + (1.0 - self._curr_supervision)  # here to avoid div by 0
            * self.unsupervised_criterion.proxy_impurity_improvement()
        )

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right,
    ) nogil:
        return self.supervised_criterion.impurity_improvement(
            impurity_parent,
            impurity_left,
            impurity_left,
        )


cdef class AxisHomogeneousCompositeSS(HomogeneousCompositeSS):
    """To use if sup and unsup criteria are AxisCriterion of the same type.
    """
    cdef void set_proxy_sup(self) nogil:
        self._proxy_sup = (
            self._curr_supervision
            * self._root_unsupervised_impurity
            * (<AxisCriterion>self.unsupervised_criterion).weighted_n_node_cols
            / (
                self._root_supervised_impurity
                * (<AxisCriterion>self.supervised_criterion).weighted_n_node_cols
            )
        )


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
            sample_indices=self.sample_indices,
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


cdef class BipartiteSemisupervisedCriterion(BipartiteCriterion):

    def __cinit__(self, *args, **kwargs):
        self._root_supervised_impurity = 0.0
        self._root_unsupervised_impurity_rows = 0.0
        self._root_unsupervised_impurity_cols = 0.0
        self._X_cols_double = None
        self._X_rows_double = None

    # TODO: improve validation
    # TODO: Regression wrapper
    def __init__(
        self,
        *,
        Criterion unsupervised_criterion_rows,
        Criterion unsupervised_criterion_cols,
        BipartiteCriterion supervised_bipartite_criterion,
        double supervision_rows,
        double supervision_cols,
        object update_supervision=None,  # callable
        SemisupervisedCriterion ss_criterion_rows=None,
        SemisupervisedCriterion ss_criterion_cols=None,
    ):
        self.supervision_rows = supervision_rows
        self.supervision_cols = supervision_cols

        self._curr_supervision_rows = supervision_rows
        self._curr_supervision_cols = supervision_cols

        self.update_supervision = update_supervision
        self._supervision_is_dynamic = update_supervision is not None

        self.unsupervised_criterion_rows = unsupervised_criterion_rows
        self.unsupervised_criterion_cols = unsupervised_criterion_cols

        self.supervised_bipartite_criterion = supervised_bipartite_criterion
        self.supervised_criterion_rows = (
            self.supervised_bipartite_criterion.criterion_rows
        )
        self.supervised_criterion_cols = (
            self.supervised_bipartite_criterion.criterion_cols
        )

        # TODO: move validation elsewhere.
        if ss_criterion_rows is None:
            self.ss_criterion_rows = SSCompositeCriterion(
                supervised_criterion=self.supervised_criterion_rows,
                unsupervised_criterion=self.unsupervised_criterion_rows,
                supervision=self._curr_supervision_rows,
            )
        else:
            self.ss_criterion_rows = ss_criterion_rows

        if ss_criterion_cols is None:
            self.ss_criterion_cols = SSCompositeCriterion(
                supervised_criterion=self.supervised_criterion_cols,
                unsupervised_criterion=self.unsupervised_criterion_cols,
                supervision=self._curr_supervision_cols,
            )
        else:
            self.ss_criterion_cols = ss_criterion_cols

        assert (
            self.supervised_criterion_rows
            is self.ss_criterion_rows.supervised_criterion
        )
        assert (
            self.unsupervised_criterion_rows
            is self.ss_criterion_rows.unsupervised_criterion
        )
        assert (
            self.supervised_criterion_cols
            is self.ss_criterion_cols.supervised_criterion
        )
        assert (
            self.unsupervised_criterion_cols
            is self.ss_criterion_cols.unsupervised_criterion
        )

    cdef void init_X(self):
        # FIXME: needed because Criterion cannot receive y as DTYPE, but
        # generates a lot (!) of unecessary memory usage in forests, as each
        # tree now hold a copy of X.
        self._X_rows_double = np.ascontiguousarray(self.X_rows, dtype='float64')
        self._X_cols_double = np.ascontiguousarray(self.X_cols, dtype='float64')

    cdef int init(
        self,
        const DTYPE_t[:, ::1] X_rows,
        const DTYPE_t[:, ::1] X_cols,
        const DOUBLE_t[:, :] y,
        const DOUBLE_t[:] row_weights,
        const DOUBLE_t[:] col_weights,
        double weighted_n_rows,
        double weighted_n_cols,
        const SIZE_t[:] row_indices,
        const SIZE_t[:] col_indices,
        SIZE_t[2] start,
        SIZE_t[2] end,
    ) nogil except -1:

        # Initialize fields
        self.X_rows = X_rows
        self.X_cols = X_cols
        self.y = y
        self.row_weights = row_weights
        self.col_weights = col_weights
        self.weighted_n_rows = weighted_n_rows
        self.weighted_n_cols = weighted_n_cols
        self.weighted_n_samples = weighted_n_rows * weighted_n_cols
        self.row_indices = row_indices
        self.col_indices = col_indices

        self.start[0] = start[0]
        self.start[1] = start[1]
        self.end[0] = end[0]
        self.end[1] = end[1]

        self.supervised_bipartite_criterion.init(
            self.X_rows,
            self.X_cols,
            self.y,
            self.row_weights,
            self.col_weights,
            self.weighted_n_rows,
            self.weighted_n_cols,
            self.row_indices,
            self.col_indices,
            self.start,
            self.end,
        )

        if self._X_rows_double is None:
            # or self._X_cols_double is None
            with gil:
                self.init_X()

        self.unsupervised_criterion_rows.init(
            y=self._X_rows_double,
            sample_weight=self.row_weights,
            weighted_n_samples=self.weighted_n_rows,
            sample_indices=self.row_indices,
            start=self.start[0],
            end=self.end[0],
        )
        self.unsupervised_criterion_cols.init(
            y=self._X_cols_double,
            sample_weight=self.col_weights,
            weighted_n_samples=self.weighted_n_cols,
            sample_indices=self.col_indices,
            start=self.start[1],
            end=self.end[1],
        )

        self.n_row_features = self.unsupervised_criterion_rows.n_outputs
        self.n_col_features = self.unsupervised_criterion_cols.n_outputs

        # Will be used by TreeBuilder as stopping criteria.
        self.weighted_n_node_rows = (
            self.supervised_bipartite_criterion.weighted_n_node_rows
        )
        self.weighted_n_node_cols = (
            self.supervised_bipartite_criterion.weighted_n_node_cols
        )

        # Will further be stored in the Tree object by the BipartiteSplitter.
        self.weighted_n_node_samples = (
            self.supervised_bipartite_criterion.weighted_n_node_samples
        )

        # Update supervision amount
        # TODO: use function pointers
        if self._supervision_is_dynamic:
            with gil:
                self._curr_supervision_rows = self.update_supervision(
                    weighted_n_samples=self.weighted_n_samples,
                    weighted_n_node_samples=self.weighted_n_node_samples,
                    weighted_n_samples_axis=self.weighted_n_rows,
                    weighted_n_node_samples_axis=self.weighted_n_node_rows,
                    current_supervision=self._curr_supervision_rows,
                    original_supervision=self.supervision_rows,
                )
                self._curr_supervision_cols = self.update_supervision(
                    weighted_n_samples=self.weighted_n_samples,
                    weighted_n_node_samples=self.weighted_n_node_samples,
                    weighted_n_samples_axis=self.weighted_n_cols,
                    weighted_n_node_samples_axis=self.weighted_n_node_cols,
                    current_supervision=self._curr_supervision_cols,
                    original_supervision=self.supervision_cols,
                )

        # FIXME: if the criteria given to the splitter is composite
        # (axis_decision_only=False), they will not update supervision
        # to use in proxy_impurity_improvement(). So we mannually set
        # them here, being the only thing requiring us to mantain
        # references to the splitters' semisupervised criterion
        # wrappers.
        # FIXME: will not work for GMO, only GSO (GSO proxy impurity needs to
        # consider both rows and column supervision).
        # FIXME: Will not work properly for different row/col sup values. Total
        # sup needs to be calculated at proxy_impurity_improvement as well.
        self.ss_criterion_rows._curr_supervision = self._curr_supervision_rows
        self.ss_criterion_cols._curr_supervision = self._curr_supervision_cols

    cdef void set_root_impurities(self) nogil:
        self.ss_criterion_rows.set_root_impurities()
        self.ss_criterion_cols.set_root_impurities()

        self._root_supervised_impurity = \
            self.supervised_bipartite_criterion.node_impurity()

        self._root_unsupervised_impurity_rows = \
            self.ss_criterion_rows._root_unsupervised_impurity
        self._root_unsupervised_impurity_cols = \
            self.ss_criterion_cols._root_unsupervised_impurity

    cdef void node_value(self, double* dest) nogil:
        self.supervised_bipartite_criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        if self._root_supervised_impurity == 0.0:
            # or root_unsupervised_impurities == 0.0
            self.set_root_impurities()
            return 1.0

        cdef:
            double s_imp, u_imp_rows, u_imp_cols
            double sup_rows, sup_cols

        sup_rows = self._curr_supervision_rows
        sup_cols = self._curr_supervision_cols

        s_imp = (
            self.supervised_bipartite_criterion.node_impurity()
            * (sup_rows + sup_cols)
            / self._root_supervised_impurity
        )
        u_imp_rows = (
            self.unsupervised_criterion_rows.node_impurity()
            * (1.0 - sup_rows)
            / self._root_unsupervised_impurity_rows
        )
        u_imp_cols = (
            self.unsupervised_criterion_cols.node_impurity()
            * (1.0 - sup_cols)
            / self._root_unsupervised_impurity_cols
        )

        return 0.5 * (u_imp_rows + u_imp_cols + s_imp)

    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    ) nogil:
        # TODO: use self.ss_criterion_* to simplify. Also, SSCompositeCriterion
        # caches children impurities.
        cdef:
            double sup, other_sup, total_sup
            double u_imp_left, u_imp_right, other_u_imp

        if axis == 0:
            other_u_imp = (
                self.unsupervised_criterion_cols.node_impurity()
                / self._root_unsupervised_impurity_cols
            )
            # TODO: remove this dependency on the way we do axis_supervision_only.
            # There is no guarantee that the splitters are using the
            # unsupervised criteria, it is a valid option to use it only for
            # choosing an axis, calculating the unsupervised impurity only
            # here. Therefore, we must ensure the unsupervised criterion is in
            # the right position.
            self.unsupervised_criterion_rows.update(
                self.supervised_criterion_rows.pos
            )
            self.unsupervised_criterion_rows.children_impurity(
                &u_imp_left, &u_imp_right,
            )
            u_imp_left /= self._root_unsupervised_impurity_rows
            u_imp_right /= self._root_unsupervised_impurity_rows

            sup = self._curr_supervision_rows
            other_sup = self._curr_supervision_cols

        elif axis == 1:
            other_u_imp = (
                self.unsupervised_criterion_rows.node_impurity()
                / self._root_unsupervised_impurity_rows
            )

            # See the previous comment.
            self.unsupervised_criterion_cols.update(
                self.supervised_criterion_cols.pos
            )
            self.unsupervised_criterion_cols.children_impurity(
                &u_imp_left, &u_imp_right,
            )
            u_imp_left /= self._root_unsupervised_impurity_cols
            u_imp_right /= self._root_unsupervised_impurity_cols

            sup = self._curr_supervision_cols
            other_sup = self._curr_supervision_rows

        else:
            with gil:
                raise InvalidAxisError

        u_imp_left = (1.0-sup) * u_imp_left + (1.0-other_sup) * other_u_imp
        u_imp_right = (1.0-sup) * u_imp_right + (1.0-other_sup) * other_u_imp

        self.supervised_bipartite_criterion.children_impurity(
            impurity_left, impurity_right, axis,
        )
        impurity_left[0] /= self._root_supervised_impurity
        impurity_right[0] /= self._root_supervised_impurity

        total_sup = sup + other_sup
        impurity_left[0] = 0.5 * (u_imp_left + total_sup * impurity_left[0])
        impurity_right[0] = 0.5 * (u_imp_right + total_sup * impurity_right[0])

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right,
        SIZE_t axis,
    ) nogil:
        if axis == 0:
            # TODO: remove this dependency on the way we do axis_supervision_only.
            # There is no guarantee that the splitters are using the
            # unsupervised criteria, it is a valid option to use it only for
            # choosing an axis, calculating the unsupervised impurity only
            # here. Therefore, we must ensure the unsupervised criterion is in
            # the right position.
            self.unsupervised_criterion_rows.update(
                self.supervised_criterion_rows.pos
            )
            return self.ss_criterion_rows.impurity_improvement(
                # SSCompositeCriterion ignores these values.
                impurity_parent,
                impurity_left,
                impurity_right,
            )
        elif axis == 1:
            # See the previous comment.
            self.unsupervised_criterion_cols.update(
                self.supervised_criterion_cols.pos
            )
            return self.ss_criterion_cols.impurity_improvement(
                # SSCompositeCriterion ignores these values.
                impurity_parent,
                impurity_left,
                impurity_right,
            )
        else:
            with gil:
                raise InvalidAxisError
