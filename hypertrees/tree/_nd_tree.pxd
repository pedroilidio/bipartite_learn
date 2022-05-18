# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

# See _tree.pyx for details.

import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from sklearn.tree._splitter cimport Splitter
from sklearn.tree._tree cimport Tree, TreeBuilder, Node

from ._nd_splitter cimport SplitRecord, Splitter2D


# =============================================================================
# Tree builder
# =============================================================================

cdef class TreeBuilderND:  # (TreeBuilder):
    # The TreeBuilder recursively builds a Tree object from training samples,
    # using a Splitter object for splitting internal nodes and assigning
    # values to leaves.
    #
    # This class controls the various stopping criteria and the node splitting
    # evaluation order, e.g. depth-first or best-first.
    #
    # The ND version adapts its _check_input method to multi-dimensional
    # splitting.

    # FIXME: All these properties should be inherited from TreeBuilder.

    cdef Splitter2D splitter             # Splitting algorithm

    cdef SIZE_t min_samples_split        # Minimum number of samples in an internal node
    cdef SIZE_t min_samples_leaf         # Minimum number of samples in a leaf
    cdef double min_weight_leaf          # Minimum weight in a leaf
    cdef SIZE_t max_depth                # Maximal tree depth
    cdef double min_impurity_decrease    # Impurity threshold for early stopping

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=*)
    cdef _check_input(self, object X, np.ndarray y,  np.ndarray sample_weight)
