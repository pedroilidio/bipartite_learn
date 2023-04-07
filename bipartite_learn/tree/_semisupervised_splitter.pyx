# cython: boundscheck=False
#
# Since splitter objects are not exported from sklearn.tree._splitter,
# this module must exist to introduce a few lines to BestSplitter.
# 
# Author: Pedro Ilidio
# Adapted from https://github.com/scikit-learn/scikit-learn/blob/1.1.1/sklearn/tree/_splitter.pyx
#
# License: BSD 3 clause

import numpy as np

from libc.stdlib cimport free
from libc.stdlib cimport qsort
from libc.string cimport memcpy
from libc.string cimport memset

from sklearn.tree._splitter cimport (
    Splitter, DTYPE_t, SIZE_t, DOUBLE_t, UINT32_t, SplitRecord,
)
from sklearn.tree._utils cimport rand_int, rand_uniform, log
from sklearn.tree._criterion cimport Criterion


cdef double INFINITY = np.inf
# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7


cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY


cdef class BaseDenseSplitter(Splitter):
    cdef const DTYPE_t[:, :] X

    cdef SIZE_t n_total_samples

    cdef int init(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  DOUBLE_t[:] sample_weight) except -1:
        """Initialize the splitter
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        # Call parent init
        Splitter.init(self, X, y, sample_weight)

        self.X = X
        return 0


cdef class BestSplitterSFSS(BaseDenseSplitter):
    """Splitter for finding the best split."""
    def __reduce__(self):
        return (BestSplitterSFSS, (self.criterion,
                               self.max_features,
                               self.min_samples_leaf,
                               self.min_weight_leaf,
                               self.random_state), self.__getstate__())

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best split on node samples[start:end]
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Find the best split
        cdef SIZE_t[::1] samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t[::1] features = self.features
        cdef SIZE_t[::1] constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t[::1] Xf = self.feature_values
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        cdef double current_proxy_improvement = -INFINITY
        cdef double best_proxy_improvement = -INFINITY

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t p
        cdef SIZE_t feature_idx_offset
        cdef SIZE_t feature_offset
        cdef SIZE_t i
        cdef SIZE_t j

        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end

        cdef double _wnns

        _init_split(&best, end)

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[
                current.feature = features[f_j]

              # ===============================================
              # bipartite_learn NOTE:                              

                # Sort samples along that feature; by
                # copying the values into an array and
                # sorting the array in a manner which utilizes the cache more
                # effectively.
                for i in range(start, end):
                    Xf[i] = self.X[samples[i], current.feature]

                sort(&Xf[start], &samples[start], end - start)

                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                # ===============================================
                # One of the two lines that we added.
                # ===============================================
                # TODO: Release GIL.                            
                # ===============================================
                    with gil:
                        self.criterion.set_feature(current.feature)
                # ===============================================

                    # Evaluate all splits
                    self.criterion.reset()
                    p = start

                    while p < end:
                        while (p + 1 < end and
                               Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                            p += 1

                        # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                        #                    X[samples[p], current.feature])
                        p += 1
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p - 1], current.feature])

                        if p < end:
                            current.pos = p

                            # Reject if min_samples_leaf is not guaranteed
                            if (((current.pos - start) < min_samples_leaf) or
                                    ((end - current.pos) < min_samples_leaf)):
                                continue

                            self.criterion.update(current.pos)

                            # Reject if min_weight_leaf is not satisfied
                            if ((self.criterion.weighted_n_left < min_weight_leaf) or
                                    (self.criterion.weighted_n_right < min_weight_leaf)):
                                continue

                            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                            if current_proxy_improvement > best_proxy_improvement:
                                best_proxy_improvement = current_proxy_improvement
                                # sum of halves is used to avoid infinite value
                                current.threshold = Xf[p - 1] / 2.0 + Xf[p] / 2.0

                                if ((current.threshold == Xf[p]) or
                                    (current.threshold == INFINITY) or
                                    (current.threshold == -INFINITY)):
                                    current.threshold = Xf[p - 1]

                                best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            partition_end = end
            p = start

            while p < partition_end:
                if self.X[samples[p], best.feature] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1

                    samples[p], samples[partition_end] = samples[partition_end], samples[p]

          # ============================================
          # bipartite_learn NOTE:
          # ============================================
            with gil:
                self.criterion.set_feature(best.feature)
          # ============================================

            self.criterion.reset()
            self.criterion.update(best.pos)
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)
            best.improvement = self.criterion.impurity_improvement(
                impurity, best.impurity_left, best.impurity_right)

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(&features[0], &constant_features[0], sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(&constant_features[n_known_constants],
               &features[n_known_constants],
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0


# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    if n == 0:
      return
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples,
        SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef DTYPE_t a = Xf[0], b = Xf[n // 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(DTYPE_t* Xf, SIZE_t *samples,
                    SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
                           SIZE_t start, SIZE_t end) nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) // 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1



cdef class RandomSplitterSFSS(BaseDenseSplitter):
    """Splitter for finding the best random split."""
    def __reduce__(self):
        return (RandomSplitterSFSS, (self.criterion,
                                     self.max_features,
                                     self.min_samples_leaf,
                                     self.min_weight_leaf,
                                     self.random_state), self.__getstate__())

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best random split on node samples[start:end]
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Draw random splits and pick the best
        cdef SIZE_t[::1] samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t[::1] features = self.features
        cdef SIZE_t[::1] constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t[::1] Xf = self.feature_values
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        cdef double current_proxy_improvement = - INFINITY
        cdef double best_proxy_improvement = - INFINITY

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t p
        cdef SIZE_t partition_end
        cdef SIZE_t feature_stride
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef SIZE_t n_visited_features = 0
        cdef DTYPE_t min_feature_value
        cdef DTYPE_t max_feature_value
        cdef DTYPE_t current_feature_value

        _init_split(&best, end)

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):
            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]

                # Find min, max
                min_feature_value = self.X[samples[start], current.feature]
                max_feature_value = min_feature_value
                Xf[start] = min_feature_value

                for p in range(start + 1, end):
                    current_feature_value = self.X[samples[p], current.feature]
                    Xf[p] = current_feature_value

                    if current_feature_value < min_feature_value:
                        min_feature_value = current_feature_value
                    elif current_feature_value > max_feature_value:
                        max_feature_value = current_feature_value

                if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
                    features[f_j], features[n_total_constants] = features[n_total_constants], current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Draw a random threshold
                    current.threshold = rand_uniform(min_feature_value,
                                                     max_feature_value,
                                                     random_state)

                    if current.threshold == max_feature_value:
                        current.threshold = min_feature_value

                    # Partition
                    p, partition_end = start, end
                    while p < partition_end:
                        if Xf[p] <= current.threshold:
                            p += 1
                        else:
                            partition_end -= 1

                            Xf[p], Xf[partition_end] = Xf[partition_end], Xf[p]
                            samples[p], samples[partition_end] = samples[partition_end], samples[p]

                    current.pos = partition_end

                    # Reject if min_samples_leaf is not guaranteed
                    if (((current.pos - start) < min_samples_leaf) or
                            ((end - current.pos) < min_samples_leaf)):
                        continue

                # ===============================================
                # bipartite_learn NOTE:                              
                # ===============================================
                # One of the two lines that we added.
                # ===============================================
                # TODO: Release GIL.                            
                # ===============================================
                    with gil:
                        self.criterion.set_feature(current.feature)
                # ===============================================

                    # Evaluate split
                    self.criterion.reset()
                    self.criterion.update(current.pos)

                    # Reject if min_weight_leaf is not satisfied
                    if ((self.criterion.weighted_n_left < min_weight_leaf) or
                            (self.criterion.weighted_n_right < min_weight_leaf)):
                        continue

                    current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            if current.feature != best.feature:
                p, partition_end = start, end

                while p < partition_end:
                    if self.X[samples[p], best.feature] <= best.threshold:
                        p += 1
                    else:
                        partition_end -= 1

                        samples[p], samples[partition_end] = samples[partition_end], samples[p]

        # ===============================================
        # bipartite_learn NOTE:                              
        # ===============================================
        # One of the two lines that we added.
        # ===============================================
        # TODO: Release GIL.                            
        # ===============================================
            with gil:
                self.criterion.set_feature(current.feature)
        # ===============================================

            self.criterion.reset()
            self.criterion.update(best.pos)
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)
            best.improvement = self.criterion.impurity_improvement(
                impurity, best.impurity_left, best.impurity_right)

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(&features[0], &constant_features[0], sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(&constant_features[n_known_constants],
               &features[n_known_constants],
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0