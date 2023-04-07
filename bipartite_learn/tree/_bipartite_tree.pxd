# Authors: Pedro Ilídio <ilidio@alumni.usp.br>
# Based on scikit-learn.
#
# License: BSD 3 clause

# See _tree.pyx for details.
from sklearn.tree._tree cimport Tree

import numpy as np
cimport numpy as cnp

from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters

from ._bipartite_splitter cimport MultipartiteSplitRecord, BipartiteSplitter


# =============================================================================
# Tree builder
# =============================================================================

# TODO: Currently we cannot inherit from sklearn.tree._tree.TreeBuilder, to
# be able to release y contiguity. and use different stopping criteria. In the
# future, we may reevaluate this possibility.
cdef class BipartiteTreeBuilder:
    # The TreeBuilder recursively builds a Tree object from training samples,
    # using a Splitter object for splitting internal nodes and assigning
    # values to leaves.
    #
    # This class controls the various stopping criteria and the node splitting
    # evaluation order, e.g. depth-first or best-first.
    #
    # The bipartite version adapts its _check_input method to multi-dimensional
    # splitting.

    cdef BipartiteSplitter splitter            # Splitting algorithm

    cdef SIZE_t min_samples_split       # Minimum number of samples in an internal node
    cdef SIZE_t min_samples_leaf        # Minimum number of samples in a leaf
    cdef double min_weight_leaf         # Minimum weight in a leaf
    cdef SIZE_t max_depth               # Maximal tree depth
    cdef double min_impurity_decrease   # Impurity threshold for early stopping

    # Bipartite parameters
    cdef SIZE_t min_rows_split          # Minimum number of rows in an internal node
    cdef SIZE_t min_rows_leaf           # Minimum number of rows in a leaf
    cdef double min_row_weight_leaf     # Minimum sum of row weights in a leaf
    cdef SIZE_t min_cols_split          # Minimum number of columns in an internal node
    cdef SIZE_t min_cols_leaf           # Minimum number of columns in a leaf
    cdef double min_col_weight_leaf     # Minimum sum of column weights in a leaf

    cpdef build(
        self,
        Tree tree,
        object X,
        const DOUBLE_t[:, :] y,
        const DOUBLE_t[:] sample_weight=*,
    )
    cdef _check_input(
        self,
        object X,
        const DOUBLE_t[:, :] y,
        const DOUBLE_t[:] sample_weight,
    )