import copy
from sklearn.tree._splitter cimport Splitter
from sklearn.tree._criterion cimport Criterion, RegressionCriterion
from sklearn.tree._criterion import MSE
from sklearn.tree._tree cimport SIZE_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset
import numpy as np
cimport numpy as cnp

from ._nd_criterion cimport RegressionCriterionWrapper2D, MSE_Wrapper2D
from ._nd_splitter import make_2d_splitter


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


cdef class RegressionCriterionWrapper2DSS(RegressionCriterionWrapper2D):
    cdef void _set_splitter_y(
        self,
        Splitter splitter,
        const DOUBLE_t[:, ::1] y,
    ):
        # FIXME: avoid using Python here.
        if splitter == self.splitter_rows:
            splitter.y = np.hstack((self.X_rows, y))
        elif splitter == self.splitter_cols:
            splitter.y = np.hstack((self.X_cols, y))

    cdef int _node_reset_child_splitter(
            self,
            Splitter child_splitter,
            const DOUBLE_t[:, ::1] y,
            DOUBLE_t* sample_weight,
            SIZE_t start,
            SIZE_t end,
            DOUBLE_t* weighted_n_node_samples,
    ) nogil except -1:
        """Substitutes splitter.node_reset() setting child splitter on 2D data.
        """
        # TODO: It is done in Splitter2D.init(). Should we do it here?
        # child_splitter.weighted_n_samples = self.weighted_n_samples
        with gil:
            self._set_splitter_y(child_splitter, y)
        # SIZE_t shape0, shape1 = y.shape[0], y.shape[1]
        # if start == 0 and end ==
        # else:
        #     child_splitter.y[:, -self.n_outputs:] = y
        child_splitter.sample_weight = sample_weight
        return child_splitter.node_reset(start, end, weighted_n_node_samples)


cdef class MSE_Wrapper2DSS(RegressionCriterionWrapper2DSS):
    cdef double impurity_improvement(
            self, double impurity_parent, double
            impurity_left, double impurity_right,
            SIZE_t axis,
    ) nogil:
        cdef double ret

        if axis == 1:
            ret = self.splitter_cols.criterion.impurity_improvement(
                impurity_parent, impurity_left, impurity_right)
        elif axis == 0:
            ret = self.splitter_rows.criterion.impurity_improvement(
                impurity_parent, impurity_left, impurity_right)

        return ret

    cdef double ss_impurity(
        self,
        double u_imp_rows,
        double u_imp_cols,
        double s_imp,
    ):
        cdef double u_imp, sup_rows, sup_cols
        cdef SIZE_t n_row_features, n_col_features

        cdef SSCompositeCriterion ss_criterion_rows = \
            self.splitter_rows.criterion
        cdef SSCompositeCriterion ss_criterion_cols = \
            self.splitter_cols.criterion
        cdef Criterion ur_criterion = \
            ss_criterion_rows.unsupervised_criterion
        cdef Criterion uc_criterion = \
            ss_criterion_cols.unsupervised_criterion

        sup_rows = ss_criterion_rows.supervision
        sup_cols = ss_criterion_cols.supervision
        n_row_features = ur_criterion.n_outputs
        n_col_features = uc_criterion.n_outputs

        u_imp_rows *= n_row_features * (1-sup_rows)
        u_imp_cols *= n_col_features * (1-sup_cols)
        u_imp = u_imp_rows + u_imp_cols
        u_imp /= n_row_features + n_col_features

        s_imp *= sup_rows + sup_cols

        # 2 = sup_rows + sup_cols + (1-sup_rows) + (1-sup_cols)
        return (u_imp + s_imp) / 2


    cdef double _node_impurity(self):
        cdef double u_imp_rows, u_imp_cols
        cdef double s_imp = MSE_Wrapper2D.node_impurity(self)

        cdef SSCompositeCriterion ss_criterion_rows = \
            self.splitter_rows.criterion
        cdef SSCompositeCriterion ss_criterion_cols = \
            self.splitter_cols.criterion
        cdef Criterion ur_criterion = \
            ss_criterion_rows.unsupervised_criterion
        cdef Criterion uc_criterion = \
            ss_criterion_cols.unsupervised_criterion

        u_imp_rows = ur_criterion.node_impurity()
        u_imp_cols = uc_criterion.node_impurity()

        return self.ss_impurity(u_imp_rows, u_imp_cols, s_imp)

    cdef double node_impurity(self) nogil:
        with gil:
            return self._node_impurity()

    cdef Criterion _get_criterion(self, SIZE_t axis): 
        cdef SSCompositeCriterion ss_criterion

        if axis == 1:
            ss_criterion = self.splitter_cols.criterion
        if axis == 0:
            ss_criterion = self.splitter_rows.criterion

        return ss_criterion.supervised_criterion

    cdef Splitter _get_splitter(self, SIZE_t axis): 
        if axis == 1:
            return self.splitter_cols
        if axis == 0:
            return self.splitter_rows

    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    ):
        cdef double s_impurity_left, s_impurity_right
        cdef double u_impurity_left, u_impurity_right
        cdef double other_u_imp

        cdef Splitter splitter = self._get_splitter(axis)
        cdef Splitter other_splitter = self._get_splitter(not axis)
        cdef SSCompositeCriterion ss_criterion, other_ss_criterion
        cdef Criterion u_crit, other_u_crit

        ss_criterion = splitter.criterion
        other_ss_criterion = other_splitter.criterion
        u_crit = ss_criterion.unsupervised_criterion
        other_u_crit = other_ss_criterion.unsupervised_criterion
        other_u_imp = other_u_crit.node_impurity()

        u_crit.children_impurity(&u_impurity_left, &u_impurity_right)

        # FIXME: the following should substitute everything until the 3rd hline
        # =====================================================================
        # MSE_Wrapper2D.children_impurity(
        #     self, &s_impurity_left, &s_impurity_right, axis)
        # =====================================================================

        cdef DOUBLE_t y_ij

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i, j, q, p, k
        cdef DOUBLE_t w = 1.0

        cdef double[::1] sum_left
        cdef double[::1] sum_right
        cdef DOUBLE_t weighted_n_left
        cdef DOUBLE_t weighted_n_right
        cdef RegressionCriterion criterion
        criterion = self._get_criterion(axis)

        cdef SIZE_t[2] end
        end[0], end[1] = self.end[0], self.end[1]

        sum_left = criterion.sum_left
        sum_right = criterion.sum_right
        weighted_n_left = criterion.weighted_n_left
        weighted_n_right = criterion.weighted_n_right
        end[axis] = criterion.pos

        with nogil:
            for p in range(self.start[0], end[0]):
                i = self.row_samples[p]
                for q in range(self.start[1], end[1]):
                    j = self.col_samples[q]

                    w = 1.0
                    if self.row_sample_weight != NULL:
                        w = self.row_sample_weight[i]
                    if self.col_sample_weight != NULL:
                        w *= self.col_sample_weight[j]

                    # TODO: multi-output
                    y_ij = self.y_2D[i, j]
                    sq_sum_left += w * y_ij * y_ij

            sq_sum_right = self.sq_sum_total - sq_sum_left

            s_impurity_left = sq_sum_left / weighted_n_left
            s_impurity_right = sq_sum_right / weighted_n_right

            for k in range(self.n_outputs):
                s_impurity_left -= (sum_left[k] / weighted_n_left) ** 2.0
                s_impurity_right -= (sum_right[k] / weighted_n_right) ** 2.0
            s_impurity_left /= self.n_outputs
            s_impurity_right /= self.n_outputs

        # =====================================================================
 
        if axis == 0:
            impurity_left[0] = self.ss_impurity(
                u_imp_rows=u_impurity_left,
                u_imp_cols=other_u_imp,
                s_imp=s_impurity_left,
            )
            impurity_right[0] = self.ss_impurity(
                u_imp_rows=u_impurity_right,
                u_imp_cols=other_u_imp,
                s_imp=s_impurity_right,
            )
        elif axis == 1:
            impurity_left[0] = self.ss_impurity(
                u_imp_rows=other_u_imp,
                u_imp_cols=u_impurity_left,
                s_imp=s_impurity_left,
            )
            impurity_right[0] = self.ss_impurity(
                u_imp_rows=other_u_imp,
                u_imp_cols=u_impurity_right,
                s_imp=s_impurity_right,
            )
            

def make_2dss_splitter(
       splitters,
       supervision=0.5,
       ss_criteria=None,
       criteria=None,
       supervised_criteria=None,
       unsupervised_criteria=None,
       n_features=None,

       n_samples=None,
       max_features=None,
       n_outputs=1,
       min_samples_leaf=1,
       min_weight_leaf=0.0,
       ax_min_samples_leaf=1,
       ax_min_weight_leaf=0.0,
       random_state=None,
       criterion_wrapper_class=MSE_Wrapper2DSS,
    ):
    """Factory function of Splitter2D instances.

    Since the building of a Splitter2D is somewhat counterintuitive, this func-
    tion is provided to simplificate the process. With exception of n_samples,
    the remaining parameters may be set to a single value or a 2-valued
    tuple or list, to specify them for each axis.
    """
    if not isinstance(n_samples, (list, tuple)):
        n_samples = [n_samples, n_samples]
    if not isinstance(n_features, (list, tuple)):
        n_features = [n_features, n_features]
    if not isinstance(n_outputs, (list, tuple)):
        n_outputs = [n_outputs, n_outputs]

    if not isinstance(supervision, (list, tuple)):
        supervision = [supervision, supervision]
    if not isinstance(ss_criteria, (list, tuple)):
        ss_criteria = [copy.deepcopy(ss_criteria) for i in range(2)]
    if not isinstance(criteria, (list, tuple)):
        criteria = [copy.deepcopy(criteria) for i in range(2)]
    if not isinstance(supervised_criteria, (list, tuple)):
        supervised_criteria = \
            [copy.deepcopy(supervised_criteria) for i in range(2)]
    if not isinstance(unsupervised_criteria, (list, tuple)):
        unsupervised_criteria = \
            [copy.deepcopy(unsupervised_criteria) for i in range(2)]

    # Make semi-supervised criteria
    for ax in range(2):
        if ss_criteria[ax] is None:
            ss_criteria[ax] = SSCompositeCriterion
        elif isinstance(ss_criteria[ax], SemisupervisedCriterion):
            criteria[ax] = copy.deepcopy(criteria[ax])
            continue
        elif isinstance(ss_criteria[ax], type):
            if not issubclass(ss_criteria[ax], SSCompositeCriterion):
                raise ValueError

        supervised_criteria[ax] = supervised_criteria[ax] or criteria[ax]
        unsupervised_criteria[ax] = unsupervised_criteria[ax] or criteria[ax]

        if isinstance(supervised_criteria[ax], type):
            if not issubclass(supervised_criteria[ax], Criterion):
                raise ValueError
            supervised_criteria[ax] = supervised_criteria[ax](
                n_outputs=n_outputs[ax],
                n_samples=n_samples[ax],
            )
        else:
            supervised_criteria[ax] = copy.deepcopy(supervised_criteria[ax])

        if isinstance(unsupervised_criteria[ax], type):
            if not issubclass(unsupervised_criteria[ax], Criterion):
                raise ValueError
            unsupervised_criteria[ax] = unsupervised_criteria[ax](
                n_outputs=n_features[ax],
                n_samples=n_samples[ax],
            )
        else:
            unsupervised_criteria[ax] = copy.deepcopy(unsupervised_criteria[ax])

        ss_criteria[ax] = ss_criteria[ax](
            supervision=supervision[ax],
            supervised_criterion=supervised_criteria[ax],
            unsupervised_criterion=unsupervised_criteria[ax],
            n_outputs=n_outputs[ax],
            n_features=n_features[ax],
            n_samples=n_samples[ax],
        )

    return make_2d_splitter(
       splitters=splitters,
       criteria=ss_criteria,  # Main change.
       n_samples=n_samples,
       max_features=max_features,
       n_outputs=n_outputs,
       min_samples_leaf=min_samples_leaf,
       min_weight_leaf=min_weight_leaf,
       ax_min_samples_leaf=ax_min_samples_leaf,
       ax_min_weight_leaf=ax_min_weight_leaf,
       random_state=random_state,
       criterion_wrapper_class=criterion_wrapper_class,
    )
