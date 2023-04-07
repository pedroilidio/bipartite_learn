## cython: boundscheck=False
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
from ._unsupervised_criterion cimport PairwiseCriterion
from ._bipartite_criterion import InvalidAxisError


cdef DOUBLE_t NAN = np.nan


cdef class SSCompositeCriterion(AxisCriterion):
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
        self.X = None
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
        self._supervised_is_axis_criterion = isinstance(
            self.supervised_criterion, AxisCriterion
        )
    
    cpdef void set_X(self, const DOUBLE_t[:, ::1] X):
        self.X = X

    cdef inline void _copy_position_wise_attributes(self) noexcept nogil:
        # NOTE: we assume the weighted_n_left/right of the supervised
        # criterion, that could sometimes diverge from the ones of the
        # unsupervised criterion.
        self.pos = self.supervised_criterion.pos
        self.weighted_n_left = self.supervised_criterion.weighted_n_left
        self.weighted_n_right = self.supervised_criterion.weighted_n_right   

    cdef int _update_supervision(self) except -1 nogil:
        # nogil to allow children classes to override it
        with gil:
            self._curr_supervision = self.update_supervision(
                y=self.y_,
                sample_indices=self.sample_indices,
                col_indices=self.col_indices,
                start=self.start,
                end=self.end,
                start_col=self.start_col,
                end_col=self.end_col,
                weighted_n_samples=self.weighted_n_samples,
                weighted_n_node_samples=self.weighted_n_node_samples,
                weighted_n_cols=self.weighted_n_cols,
                weighted_n_node_cols=self.weighted_n_node_cols,
                current_supervision=self._curr_supervision,
                original_supervision=self.supervision,
            )

    cdef void set_root_impurities(self) nogil:
        self._root_supervised_impurity = (
            self.supervised_criterion.node_impurity()
        )
        self._root_unsupervised_impurity = (
            self.unsupervised_criterion.node_impurity()
        )

    cdef int init(
        self, const DOUBLE_t[:, ::1] y,
        DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        SIZE_t[:] sample_indices,
        SIZE_t start,
        SIZE_t end,
    ) nogil except -1:
        if self.X is None:
            with gil:
                raise RuntimeError("You must set_X() before init()")
        self._cached_pos = SIZE_MAX  # Important to reset cached values.
        self.y = y
        self.y_ = y  # to be used by self._update_supervision()
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        
        self.supervised_criterion.init(
            self.y,
            sample_weight,
            weighted_n_samples,
            sample_indices,
            start,
            end,
        )
        self.unsupervised_criterion.init(
            self.X,
            sample_weight,
            weighted_n_samples,
            sample_indices,
            start,
            end,
        )

        self.weighted_n_node_samples = \
            self.supervised_criterion.weighted_n_node_samples

        if self.update_supervision is not None:
            self._update_supervision()

        if self._root_supervised_impurity == 0.0:
            # Root unupervised impurities == 0.0 as well
            self.set_root_impurities()
        
        self._copy_position_wise_attributes()  # Because criteria were reset
        self._set_proxy_supervision()

        return 0

    cdef int axis_init(
        self,
        const DOUBLE_t[:, :] y,
        const DOUBLE_t[:] sample_weight,
        const DOUBLE_t[:] col_weights,
        const SIZE_t[:] sample_indices,
        const SIZE_t[:] col_indices,
        double weighted_n_samples,
        double weighted_n_cols,
        SIZE_t start,
        SIZE_t end,
        SIZE_t start_col,
        SIZE_t end_col,
    ) except -1 nogil:
        if self.X is None:
            with gil:
                raise RuntimeError("You must set_X() before init()")
        if not self._supervised_is_axis_criterion:
            with gil:
                raise TypeError("supervised_criterion is not an AxisCriterion")

        self._cached_pos = SIZE_MAX  # Important to reset cached values.
        # Initialize fields
        self.y_ = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.n_node_cols = end_col - start_col
        self.weighted_n_samples = weighted_n_samples

        # We do not call init_columns(), so we make these attributions here
        self.col_weights = col_weights
        self.weighted_n_cols = weighted_n_cols
        self.col_indices = col_indices
        self.start_col = start_col
        self.end_col = end_col

        (<AxisCriterion>self.supervised_criterion).axis_init(
            y,
            sample_weight=sample_weight,
            col_weights=col_weights,
            weighted_n_samples=weighted_n_samples,
            weighted_n_cols=weighted_n_cols,
            sample_indices=sample_indices,
            col_indices=col_indices,
            start=start,
            end=end,
            start_col=start_col,
            end_col=end_col,
        )

        self._columns_are_set = True  # Otherwise error would have been raised

        # We do not call self.init_columns(), but instead copy attributes from
        # the child criterion.
        self._node_col_indices = \
            (<AxisCriterion>self.supervised_criterion)._node_col_indices
        self.weighted_n_cols = \
            (<AxisCriterion>self.supervised_criterion).weighted_n_cols

        self.unsupervised_criterion.init(
            self.X,
            sample_weight,
            weighted_n_samples,
            sample_indices,
            start,
            end,
        )

        self.weighted_n_node_samples = \
            self.supervised_criterion.weighted_n_node_samples

        if self.update_supervision is not None:
            self._update_supervision()

        if self._root_supervised_impurity == 0.0:
            # Root unupervised impurities == 0.0 as well
            self.set_root_impurities()
        
        self._copy_position_wise_attributes()  # Because criteria were reset
        self._set_proxy_supervision()

        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criteria at pos=start."""
        self.supervised_criterion.reset()
        self.unsupervised_criterion.reset()
        self._copy_position_wise_attributes()
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criteria at pos=end."""
        self.supervised_criterion.reverse_reset()
        self.unsupervised_criterion.reverse_reset()
        self._copy_position_wise_attributes()
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
        self._copy_position_wise_attributes()
        return 0

    cdef double node_impurity(self) nogil:
        """Calculate the impurity of the node.
        Impurity of the current node, i.e. the impurity of sample_indices[start:end].
        This is the primary function of the criterion class. The smaller the
        impurity the better.
        """
        # FIXME: init is called before node_impurity by the TreeBuilder, where
        # root impurities are also set. We will redundantly calculate the root
        # impurity here as a consequence, divind it by itself to get 1 again.
        if self._root_supervised_impurity == 0.0:
            # Root unupervised impurities == 0.0 as well
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
            + (1.0 - sup) * u_impurity_left / self._root_unsupervised_impurity
        )
        impurity_right[0] = (
            sup * s_impurity_right / self._root_supervised_impurity
            + (1.0 - sup) * u_impurity_right / self._root_unsupervised_impurity
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

    cdef void total_node_value(self, double* dest) nogil:
        """Compute a single node value for all targets, disregarding y's shape.

        This method is used instead of node_value() in cases where the
        different columns of y are *not* considered as different outputs, being
        usually equivalent to node_value if y were to be flattened, i.e.

            total_node_value(y) == node_value(y.reshape[-1, 1])

        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """
        if self._supervised_is_axis_criterion:
            (<AxisCriterion>self.supervised_criterion).total_node_value(dest)
        else:
            with gil:
                raise TypeError(
                    "total value only available for supervised AxisCriterion"
                ) 

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right,
    ) nogil:
        """Compute the improvement in impurity.

        impurity_improvement = (
            self._curr_supervision
            * self.supervised_criterion.node_impurity()
            / self._root_supervised_impurity

            + (1 - self._curr_supervision)
            * self.unsupervised_criterion.node_impurity()
            / self._root_unsupervised_impurity
        )
        """
        # FIXME: Since we cannot access s_impurity and u_impurity
        # separately, we have to discard this method's input and calculate
        # them again. To mitigate this problem, self.ss_children_impurities()
        # caches the previous values it calculated and reuses them if
        # self.pos == self._cached_pos. If both criteria's impurity improvement
        # were the same linear combination of children impurities this would
        # not be a problem (as is the case with almost all sklearn Criterion
        # classes), but criteria such as FriedmanMSE would behave differently,
        # and we try to be as general as possible. Receiving a custom split
        # record instead of each impurity seems the ideal solution, since we
        # would be able to store and pass along the supervised and unsupervised
        # impurities separately.
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
        ) / self._root_unsupervised_impurity

        s_improvement = self.supervised_criterion.impurity_improvement(
            s_impurity_parent, s_impurity_left, s_impurity_right,
        ) / self._root_supervised_impurity

        return sup * s_improvement + (1.0 - sup) * u_improvement

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        return (
            self._proxy_supervision
            * self.supervised_criterion.proxy_impurity_improvement()
            + (1.0 - self._curr_supervision)  # here to avoid div by 0
            * self.unsupervised_criterion.proxy_impurity_improvement()
        )

    cdef int _set_proxy_supervision(self) except -1 nogil:
        """Combine all constants weighting the proxy impurities.

        Avoids performing these operations at every split position in
        the method proxy_impurity_improvement, performing them only once in
        init or axis_init.
        """
        self._proxy_supervision = (
            self._curr_supervision
            * self.unsupervised_criterion._proxy_improvement_factor()
            / self.supervised_criterion._proxy_improvement_factor()
            * self._root_unsupervised_impurity
            / self._root_supervised_impurity
        )

    cdef double _proxy_improvement_factor(self) noexcept nogil:
        """If improvement = proxy_improvement / a + b, this method returns a.

        This is useful when defining proxy impurity improvements for
        compositions of Criterion objects.
        """
        return (
            self.unsupervised_criterion._proxy_improvement_factor()
            * self._root_unsupervised_impurity
        )


# =============================================================================
# Bipartite Semi-supervised Criterion Wrapper
# =============================================================================


cdef class BipartiteSemisupervisedCriterion(GMO):

    def __cinit__(self, *args, **kwargs):
        self._root_supervised_impurity = 0.0
        self._root_unsupervised_impurity_rows = 0.0
        self._root_unsupervised_impurity_cols = 0.0

    def __init__(self, BipartiteCriterion bipartite_criterion):
        self.bipartite_criterion = bipartite_criterion
        self.criterion_rows = bipartite_criterion.criterion_rows
        self.criterion_cols = bipartite_criterion.criterion_cols

    cpdef void set_X(
        self,
        const DOUBLE_t[:, ::1] X_rows,
        const DOUBLE_t[:, ::1] X_cols,
    ):
        self.criterion_rows.set_X(X_rows)
        self.criterion_cols.set_X(X_cols)

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

        self.bipartite_criterion.init(
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

        self.n_row_features = (<SSCompositeCriterion>self.criterion_rows).n_features
        self.n_col_features = (<SSCompositeCriterion>self.criterion_cols).n_features

        # Will be used by TreeBuilder as stopping criteria.
        self.weighted_n_node_rows = (
            self.bipartite_criterion.weighted_n_node_rows
        )
        self.weighted_n_node_cols = (
            self.bipartite_criterion.weighted_n_node_cols
        )

        # Will further be stored in the Tree object by the BipartiteSplitter.
        self.weighted_n_node_samples = (
            self.bipartite_criterion.weighted_n_node_samples
        )

        cdef double total_sup

        self._curr_supervision_rows = \
            (<SSCompositeCriterion>self.criterion_rows)._curr_supervision
        self._curr_supervision_cols = \
            (<SSCompositeCriterion>self.criterion_cols)._curr_supervision

        total_sup = (
            self._curr_supervision_rows + self._curr_supervision_cols
        )

        (<SSCompositeCriterion>self.criterion_rows)._curr_supervision = \
            total_sup / (self._curr_supervision_cols + 1)
        (<SSCompositeCriterion>self.criterion_cols)._curr_supervision = \
            total_sup / (self._curr_supervision_rows + 1)

        (<SSCompositeCriterion>self.criterion_rows)._set_proxy_supervision()
        (<SSCompositeCriterion>self.criterion_cols)._set_proxy_supervision()

    cdef void node_value(self, double* dest) nogil:
        self.bipartite_criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        return 0.5 * (
            self.criterion_rows.node_impurity()
            + self.criterion_cols.node_impurity()
        )

    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    ) nogil:
        cdef:
            void* ss_criterion_ptr = self._get_criterion(axis)
            void* other_ss_criterion_ptr = self._get_criterion(1 - axis)
            void* u_other_criterion_ptr
            double other_sup
            double scaled_u_other_imp
            double u_other_root_imp

        u_other_criterion_ptr = <void*>(
            (<SSCompositeCriterion>other_ss_criterion_ptr)
            .unsupervised_criterion
        )
        u_other_root_imp = (
            (<SSCompositeCriterion>other_ss_criterion_ptr)
            ._root_unsupervised_impurity
        )
        if axis == 1:
            other_sup = self._curr_supervision_rows
        elif axis == 0:
            other_sup = self._curr_supervision_cols
        else:
            with gil:
                raise InvalidAxisError(axis)

        # There is no guarantee that the splitters are using the
        # unsupervised criteria, it is a valid option to use it only for
        # choosing an axis, calculating the unsupervised impurity only
        # here. Therefore, we must ensure the unsupervised criterion is in
        # the right position.
        # TODO: remove this dependency on the way we do axis_supervision_only.
        (<SSCompositeCriterion>ss_criterion_ptr).update(
            (<SSCompositeCriterion>ss_criterion_ptr).supervised_criterion.pos
        )

        scaled_u_other_imp = (
            (<Criterion>u_other_criterion_ptr).node_impurity()
            * (0.5 - 0.5 * other_sup)
            / u_other_root_imp
        )

        (<SSCompositeCriterion>ss_criterion_ptr).children_impurity(
            impurity_left,
            impurity_right,
        )

        impurity_left[0] = (
            (0.5 + 0.5 * other_sup) * impurity_left[0]
            + scaled_u_other_imp
        )
        impurity_right[0] = (
            (0.5 + 0.5 * other_sup) * impurity_right[0]
            + scaled_u_other_imp
        )

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right,
        SIZE_t axis,
    ) nogil:
        cdef void* criterion_ptr = self._get_criterion(axis)
        cdef double other_sup

        if axis == 0:
            other_sup = self._curr_supervision_cols
        elif axis == 1:
            other_sup = self._curr_supervision_rows
        else:
            with gil:
                raise InvalidAxisError(axis)

        # There is no guarantee that the splitters are using the
        # unsupervised criteria, it is a valid option to use it only for
        # choosing an axis, calculating the unsupervised impurity only
        # here. Therefore, we must ensure the unsupervised criterion is in
        # the right position.
        # TODO: remove this dependency on the way we do axis_supervision_only.
        (<SSCompositeCriterion>criterion_ptr).update(
            (<SSCompositeCriterion>criterion_ptr).supervised_criterion.pos
        )
        return (
            (<SSCompositeCriterion>criterion_ptr).impurity_improvement(
                impurity_parent, impurity_left, impurity_right
            )
            * (0.5 * other_sup + 0.5)
        )
