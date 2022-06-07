# distutils: language = c++
from cpython cimport Py_INCREF, PyObject, PyTypeObject
from libc.stdint cimport SIZE_MAX

import struct

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)
    int PyArray_SetBaseObject(np.ndarray arr, PyObject* obj)

cdef extern from "<stack>" namespace "std" nogil:
    cdef cppclass stack[T]:
        ctypedef T value_type
        stack() except +
        bint empty()
        void pop()
        void push(T&) except +  # Raise c++ exception for bad_alloc -> MemoryError
        T& top()

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps

# Some handy constants (BestFirstTreeBuilder)
cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

# =============================================================================
# TreeBuilderND
# =============================================================================

cdef class TreeBuilderND:  # (TreeBuilder):
    """Interface for different tree building strategies."""

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None):
        """Build a decision tree from the training set (X, y)."""
        # TODO: should be inherited.

    cdef inline _check_input(self, object X, np.ndarray y,
                             np.ndarray sample_weight):
        """Check input dtype, layout and format.

        Applies the same processing as sklearn's TreeBuilder for each X[ax]
        feature matrix in X.
        """
        for ax in range(len(X)):
            if issparse(X[ax]):
                X[ax] = X[ax].tocsc()
                X[ax].sort_indices()

                if X[ax].data.dtype != DTYPE:
                    X[ax].data = np.ascontiguousarray(X[ax].data, dtype=DTYPE)

                if X[ax].indices.dtype != np.int32 or \
                   X[ax].indptr.dtype != np.int32:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

            elif X[ax].dtype != DTYPE:
                # since we have to copy we will make it fortran for efficiency
                X[ax] = np.asfortranarray(X[ax], dtype=DTYPE)

        if (sample_weight is not None and
           (sample_weight.dtype != DOUBLE or
           not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight,
                                           dtype=DOUBLE,
                                           order="C")

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        return X, y, sample_weight


# Depth first builder ---------------------------------------------------------
# A record on the stack for depth-first tree growing
cdef struct StackRecord2D:
    SIZE_t start_row
    SIZE_t start_col
    SIZE_t end_row
    SIZE_t end_col
    SIZE_t depth
    SIZE_t parent
    bint is_left
    double impurity
    SIZE_t n_constant_row_features
    SIZE_t n_constant_col_features

cdef class DepthFirstTreeBuilder2D(TreeBuilderND):
    """Build a decision tree in depth-first fashion, from 2D training data.

    It adds minor changes to sklearn's DepthfirstTreeBuilder, essentially
    in storing two-values start/end node positions and managing getting the
    new ones after each split.

    `X` is now a list with 2 matrices, for the feature table of row and column
    instances respectively. `y` is a 2 dimensional ndarray representing labels
    of interaction of each row with each column. Mulit-output is not yet imp-
    lemented.
    """
    # TODO: define separate methods for split -> node data conversion and
    # evaluating stopping criteria.
    # TODO: define axis-specific min_samples_leaf, min_samples_split and
    # min_weight_leaf, turning them into arrays. A complication is that
    # __cinit__ does not take arrays (pointers) as arguments, only python
    # objects.
    def __cinit__(self, Splitter2D splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, double min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
    ):
        """Build a decision tree from the training set (X, y)."""
        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
             sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef Splitter2D splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease

        # Recursive partition (without actual recursion)
        # TODO: test sample_weight
        splitter.init(X, y, sample_weight_ptr)

        cdef SIZE_t[2] start, start_left, start_right
        cdef SIZE_t[2] end, end_left, end_right
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_node_samples
        cdef SplitRecord split
        cdef SIZE_t node_id

        cdef double impurity = INFINITY
        cdef SIZE_t[2] n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef stack[StackRecord2D] builder_stack
        cdef StackRecord2D stack_record

        # FIXME: stack doesn't receive array elements without GIL. Would be nice
        # to have a DOUBLE_t[2] start or end, for instance.
        with nogil:
            # push root node onto stack
            builder_stack.push({
                "start_row": 0,
                "start_col": 0,
                "end_row": splitter.splitter_rows.n_samples,
                "end_col": splitter.splitter_cols.n_samples,
                "depth": 0,
                "parent": _TREE_UNDEFINED,
                "is_left": 0,
                "impurity": INFINITY,
                "n_constant_row_features": 0,
                "n_constant_col_features": 0,
            })

            while not builder_stack.empty():
                stack_record = builder_stack.top()
                builder_stack.pop()

                start[0] = stack_record.start_row
                start[1] = stack_record.start_col
                end[0] = stack_record.end_row
                end[1] = stack_record.end_col
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features[0] = stack_record.n_constant_row_features
                n_constant_features[1] = stack_record.n_constant_col_features

                n_node_samples = (end[0]-start[0]) * (end[1]-start[1])
                splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = depth >= max_depth

                is_leaf = (is_leaf or n_node_samples < min_samples_split
                           or n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                # impurity == 0 with tolerance due to rounding errors
                is_leaf = is_leaf or impurity <= EPSILON

                if not is_leaf:
                    splitter.node_split(impurity, &split, n_constant_features)
                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (split.pos >= end[split.axis] or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                node_id = tree._add_node(parent, is_left, is_leaf,
                                         split.feature,
                                         split.threshold, impurity,
                                         n_node_samples,
                                         weighted_n_node_samples)

                if node_id == SIZE_MAX:
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                splitter.node_value(tree.value + node_id * tree.value_stride)

                if not is_leaf:
                    # FIXME: review.
                    ## https://github.com/cython/cython/pull/1663
                    # start_left[:] = start[:]
                    # end_left[:] = end[:]

                    # for ax in range(n_dimensions):
                    #     start_left[ax] = start[ax]
                    #     start_right[ax] = start[ax]
                    #     end_left[ax], end_right[ax] = end[ax], end[ax]

                    start_left[0], start_right[0] = start[0], start[0]
                    start_left[1], start_right[1] = start[1], start[1]
                    end_left[0], end_right[0] = end[0], end[0]
                    end_left[1], end_right[1] = end[1], end[1]

                    # Setting new nodes coordinates.
                    # These lines are why we had to rewrite this whole method
                    # to develop the 2-dimensional version.
                    start_right[split.axis] = split.pos
                    end_left[split.axis] = split.pos

                    # Push right child on stack
                    builder_stack.push({
                        "start_row": start_right[0],
                        "start_col": start_right[1],
                        "end_row": end_right[0],
                        "end_col": end_right[1],
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 0,
                        "impurity": split.impurity_right,
                        "n_constant_row_features": n_constant_features[0],
                        "n_constant_col_features": n_constant_features[1],
                    })

                    # Push left child on stack
                    builder_stack.push({
                        "start_row": start_left[0],
                        "start_col": start_left[1],
                        "end_row": end_left[0],
                        "end_col": end_left[1],
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 1,
                        "impurity": split.impurity_left,
                        "n_constant_row_features": n_constant_features[0],
                        "n_constant_col_features": n_constant_features[1],
                    })

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()


# Best first builder ----------------------------------------------------------

cdef class BestFirstTreeBuilderND(TreeBuilderND):
    pass # TODO
