# distutils: language = c++
from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.math cimport fabs
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX

import struct

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse
from scipy.sparse import csr_matrix

from sklearn.tree._utils cimport PriorityHeap
from sklearn.tree._utils cimport PriorityHeapRecord
from sklearn.tree._utils cimport safe_realloc
from sklearn.tree._utils cimport sizet_ptr_to_ndarray

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

##################### debug
from pprint import pprint
import matplotlib.pyplot as plt

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

# # Build the corresponding numpy dtype for Node.
# # This works by casting `dummy` to an array of Node of length 1, which numpy
# # can construct a `dtype`-object for. See https://stackoverflow.com/q/62448946
# # for a more detailed explanation.
# cdef Node dummy;
# NODE_DTYPE = np.asarray(<Node[:1]>(&dummy)).dtype

# from sklearn.tree._tree import (
#     DTYPE, DOUBLE, INFINITY, EPSILON, IS_FIRST, IS_NOT_FIRST, IS_LEFT,
#     IS_NOT_LEFT, TREE_LEAF, TREE_UNDEFINED, _TREE_LEAF, _TREE_UNDEFINED,
#     INITIAL_STACK_SIZE, NODE_DTYPE,
# )
from _nd_splitter cimport SplitRecord, Splitter2D

# =============================================================================
# TreeBuilder
# =============================================================================

cdef class TreeBuilderND:
    """Interface for different tree building strategies."""

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None):
        """Build a decision tree from the training set (X, y)."""
        pass

    cdef inline _check_input(self, object X, np.ndarray y,
                             np.ndarray sample_weight):
        """Check input dtype, layout and format"""
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
                (sample_weight[ax].dtype != DOUBLE or
                not sample_weight[ax].flags.contiguous)):
                    sample_weight[ax] = np.asarray(sample_weight[ax],
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
    """Build a 2D decision tree in depth-first fashion.

    It adds minor changes to sklearn's DepthfirstTreeBuilder, essentially
    in storing two-values start/end positions and managing getting the new ones.
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
        DEBUG_TREE_MAP = np.zeros(y.astype(int).shape, dtype=float)

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef (DOUBLE_t*)[2] sample_weight_ptr = [NULL, NULL]
        # TODO
        # if sample_weight is not None:
        #     sample_weight_ptr = <(DOUBLE_t*)[2]> sample_weight.data

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
        cdef double weighted_n_samples = splitter.weighted_n_samples
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
        cdef SIZE_t ax

        # FIXME: stack doesn't receive array elements without GIL.
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

                # n_node_samples[0] = end[0] - start[0]
                # n_node_samples[1] = end[1] - start[1]
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
                    is_leaf = (is_leaf or split.pos >= end[split.axis] or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                node_id = tree._add_node(parent, is_left, is_leaf,
                                         split.feature,
                                         split.threshold, impurity,
                                         # TODO: for each axis:
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

                    start_right[split.axis] = split.pos
                    end_left[split.axis] = split.pos

                    ##### debug
                    # with gil:
                    #     pprint({
                    #         "right_start": start_right,
                    #         "right_end": end_right,
                    #         "left_start": start_left,
                    #         "left_end": end_left,
                    #     })
                    #     pprint(split)
                    #     pprint(stack_record)
                    #     print('===================')

                        #DEBUG_TREE_MAP[
                        #    start_right[0]:end_right[0],
                        #    start_right[1]:end_right[1]
                        #    ] += 1./(depth+1)
                        #plt.cla()
                        #plt.pcolormesh(DEBUG_TREE_MAP)
                        ## Remember i is y, j is x
                        #plt.plot(start_right[1], start_right[0], 'ro', mec='k')
                        #plt.plot(end_right[1], end_right[0], 'rD', mec='k')
                        #plt.plot(start_left[1], start_left[0], 'go', mec='k')
                        #plt.plot(end_left[1], end_left[0], 'gD', mec='k')
                        #plt.show()

                    ######################
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
