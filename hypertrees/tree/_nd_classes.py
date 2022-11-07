"""
This module gathers tree-based methods, including decision, regression and
randomized trees, adapted from sklearn for 2D training data.
"""

# Author: Pedro Ilidio <pedrilidio@gmail.com>
# Adapted from scikit-learn.
#
# License: BSD 3 clause


import numbers
import warnings
import copy
from abc import ABCMeta
from abc import abstractmethod
from itertools import product
from math import ceil
from typing import Iterable, Callable

import numpy as np
from scipy.sparse import issparse

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import is_classifier
from sklearn.base import MultiOutputMixin
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets

from sklearn.tree._criterion import Criterion
from sklearn.tree._splitter import Splitter
from sklearn.tree._tree import Tree
from sklearn.tree._classes import (
    DENSE_SPLITTERS, CRITERIA_CLF, CRITERIA_REG, DecisionTreeRegressor,
    DecisionTreeClassifier,
)
from sklearn.tree._tree import DTYPE, DOUBLE

from sklearn.tree._classes import BaseDecisionTree
from ..base import BaseMultipartiteEstimator, MultipartiteRegressorMixin
from ._nd_tree import DepthFirstTreeBuilder2D
from ._nd_criterion import (
    CriterionWrapper2D, MSE_Wrapper2D, PBCTCriterionWrapper,
)
from ._nd_splitter import Splitter2D, make_2d_splitter
from ..melter import row_cartesian_product


__all__ = [
    "DecisionTreeRegressor2D",
    "ExtraTreeRegressor2D",
    "BiclusteringTreeRegressor",
    "BiclusteringTreeClassifier",
    "PBCT",
]


# =============================================================================
# Types and constants
# =============================================================================

CRITERIA_2D = {
    "squared_error": MSE_Wrapper2D,
    "local_multioutput": PBCTCriterionWrapper,
}

SPARSE_SPLITTERS = {}


# =============================================================================
# Base ND decision tree
# =============================================================================


class BaseDecisionTree2D(BaseMultipartiteEstimator, BaseDecisionTree,
                         metaclass=ABCMeta):
    """Base class for ND trees, 2D adapted.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(
        self,
        *,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,

        max_leaf_nodes,
        random_state,
        min_impurity_decrease,
        class_weight=None,
        ccp_alpha=0.0,

        # 2D parameters:
        # TODO: ax_min_samples_split,
        ax_min_samples_leaf=1,
        ax_min_weight_fraction_leaf=None,
        ax_max_features=None,
        criterion_wrapper=None,
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features

        # 2D parameters:
        self.ax_min_samples_leaf = ax_min_samples_leaf
        self.ax_min_weight_fraction_leaf = ax_min_weight_fraction_leaf
        self.ax_max_features = ax_max_features
        self.criterion_wrapper = criterion_wrapper

        self.max_leaf_nodes = max_leaf_nodes

        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y, sample_weight=None, check_input=True):

        random_state = check_random_state(self.random_state)

        check_scalar(
            self.ccp_alpha,
            name="ccp_alpha",
            target_type=numbers.Real,
            min_val=0.0,
        )


        if check_input:
            # SK: Need to validate separately here.
            # SK: We can't pass multi_ouput=True because that would allow y
            # to be csr.
            # FIXME: set multi_output=False
            check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
            check_y_params = dict(multi_output=True)

            y = self._validate_data(X="no_validation", y=y, **check_y_params)
            X = self._validate_data(X, **check_X_params)

            for ax in range(len(X)):
                if issparse(X[ax]):
                    X[ax].sort_indices()

                    if (X[ax].indices.dtype != np.intc
                            or X[ax].indptr.dtype != np.intc):
                        raise ValueError(
                            "No support for np.int64 index based sparse matrices"
                        )

            if self.criterion == "poisson":
                if np.any(y < 0):
                    raise ValueError(
                        "Some value(s) of y are negative which is"
                        " not allowed for Poisson regression."
                    )
                if np.sum(y) <= 0:
                    raise ValueError(
                        "Sum of y is not positive which is "
                        "necessary for Poisson regression."
                    )

        # Determine output settings
        # NOTE: n_features_in_ must be set after calling self._validate_data.
        #       Otherwise, the method will try to compare self.n_features_in_
        #       to X[ax].shape[1] and throw an error when they do not match.
        self.ax_n_features_in_ = [Xax.shape[1] for Xax in X]
        n_samples, self.n_features_in_ = y.size, sum(self.ax_n_features_in_)
        is_classification = is_classifier(self)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        # self.n_outputs_ = y.shape[-1]  # TODO: implement multi-output (3D y).
        self.n_outputs_ = self._get_n_outputs(X, y)

        if is_classification:
            raise NotImplementedError(
                "Let's not talk about classification for now.")
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            y_encoded = np.zeros(y.shape, dtype=int)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
            y = y_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original
                )

            self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        (
            max_depth,
            min_samples_leaf,
            min_samples_split,
            max_features,
            # self.max_features_ = max_features
            max_leaf_nodes,
            sample_weight,
            min_weight_leaf
        ) = self._check_parameters(
            X, y, sample_weight, expanded_class_weight, n_samples)

        # TODO: move to _check_parameters.
        if self.ax_max_features is None:
            ax_max_features = self.ax_n_features_in_
        else:
            ax_max_features = self.ax_max_features

        ax_min_samples_leaf = self.ax_min_samples_leaf

        # TODO: move to _check_parameters.
        if self.ax_min_weight_fraction_leaf is None:
            # ax_min_weight_fraction_leaf = 0.0
            ax_min_weight_leaf = (0.0, 0.0)
        elif sample_weight is None:
            ax_min_weight_leaf = [
                mw * d for mw, d in
                zip(self.ax_min_weight_fraction_leaf, y.shape)
            ]
        else:
            warnings.warn("sample_weights is still an experimental feature.")
            split_indices = np.cumsum(y.shape)
            ax_sample_weight = np.split(sample_weight, split_indices)[:-1]
            weighted_n_samples = np.prod([
                np.sum(sw) for sw in _ax_sample_weight])
            ax_min_weight_leaf = [
                mw * weighted_n_samples for mw in
                self.ax_min_weight_fraction_leaf
            ]

        if is_classifier(self):
            self.tree_ = Tree(self.n_features_in_, self.n_classes_, self.n_outputs_)
        else:
            self.tree_ = Tree(
                self.n_features_in_,
                # TODO: tree shouldn't need this in this case
                np.array([1] * self.n_outputs_, dtype=np.intp),
                self.n_outputs_,
            )

        splitter = self._make_splitter(
            n_samples=y.shape,
            sparse=issparse(X),
            ax_max_features=ax_max_features,
            min_samples_leaf=min_samples_leaf,
            min_weight_leaf=min_weight_leaf,
            ax_min_samples_leaf=ax_min_samples_leaf,
            ax_min_weight_leaf=ax_min_weight_leaf,
            random_state=random_state,
        )

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder2D(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
            raise NotImplementedError("Use max_leaf_nodes=None")
            builder = BestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )

        builder.build(self.tree_, X, y, sample_weight)

        if self.n_outputs_ == 1 and is_classifier(self):
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        self._prune_tree()

        return self
    
    def _make_splitter(
        self,
        n_samples,
        sparse=False,
        ax_max_features=None,
        min_samples_leaf=1,
        min_weight_leaf=0.,
        ax_min_samples_leaf=1,
        ax_min_weight_leaf=0.,
        random_state=None,
    ):
        if isinstance(self.splitter, Splitter2D):
            return self.splitter

        criterion_wrapper = self.criterion_wrapper
        if isinstance(criterion_wrapper, str):
            criterion_wrapper = CRITERIA_2D[criterion_wrapper]

        criterion = self.criterion
        if not isinstance(criterion, (tuple, list)):
            criterion = [criterion, criterion]
        for ax in range(2):
            if isinstance(criterion[ax], str):
                if is_classifier(self):
                    criterion[ax] = CRITERIA_CLF[criterion[ax]]
                else:
                    criterion[ax] = CRITERIA_REG[criterion[ax]]

        SPLITTERS = SPARSE_SPLITTERS if sparse else DENSE_SPLITTERS

        # NOTE: make_2d_splitter takes charge of that.
        # Make a deepcopy in case the criterion has mutable attributes that
        # might be shared and modified concurrently during parallel fitting
        # criterion = copy.deepcopy(criterion)
        splitter = self.splitter
        if not isinstance(splitter, (tuple, list)):
            splitter = [splitter, splitter]
        for ax in range(2):
            if isinstance(splitter[ax], str):
                splitter[ax] = SPLITTERS[splitter[ax]]

        splitter = make_2d_splitter(
            splitters=splitter,
            criteria=criterion,
            n_samples=n_samples,
            n_outputs=self.n_outputs_,
            # TODO: check ax_* parameters.
            max_features=ax_max_features,
            min_samples_leaf=min_samples_leaf,
            min_weight_leaf=min_weight_leaf,
            ax_min_samples_leaf=ax_min_samples_leaf,
            ax_min_weight_leaf=ax_min_weight_leaf,
            random_state=random_state,
            criterion_wrapper_class=criterion_wrapper,
        )

        return splitter

    def _check_parameters(self, X, y, sample_weight, expanded_class_weight, n_samples):
        if self.max_depth is not None:
            check_scalar(
                self.max_depth,
                name="max_depth",
                target_type=numbers.Integral,
                min_val=1,
            )
        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth

        if isinstance(self.min_samples_leaf, numbers.Integral):
            check_scalar(
                self.min_samples_leaf,
                name="min_samples_leaf",
                target_type=numbers.Integral,
                min_val=1,
            )
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            check_scalar(
                self.min_samples_leaf,
                name="min_samples_leaf",
                target_type=numbers.Real,
                min_val=0.0,
                include_boundaries="neither",
            )
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            check_scalar(
                self.min_samples_split,
                name="min_samples_split",
                target_type=numbers.Integral,
                min_val=2,
            )
            min_samples_split = self.min_samples_split
        else:  # float
            check_scalar(
                self.min_samples_split,
                name="min_samples_split",
                target_type=numbers.Real,
                min_val=0.0,
                max_val=1.0,
                include_boundaries="right",
            )
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        check_scalar(
            self.min_weight_fraction_leaf,
            name="min_weight_fraction_leaf",
            target_type=numbers.Real,
            min_val=0.0,
            max_val=0.5,
        )

        axes_n_features = [Xax.shape[1] for Xax in X]

        if not isinstance(self.max_features, (tuple, list)):
            max_features = [self.max_features, self.max_features]
        for i in range(2):
            if isinstance(max_features[i], str):
                if max_features[i] == "auto":
                    if is_classification:
                        max_features[i] = max(1, int(np.sqrt(axes_n_features[i])))
                    else:
                        max_features[i] = self.n_features_in_[i]
                elif max_features[i] == "sqrt":
                    max_features[i] = max(1, int(np.sqrt(axes_n_features[i])))
                elif max_features[i] == "log2":
                    max_features[i] = max(1, int(np.log2(axes_n_features[i])))
                else:
                    raise ValueError(
                        "Invalid value for max_features. "
                        "Allowed string values are 'auto', "
                        "'sqrt' or 'log2'."
                    )
            elif max_features[i] is None:
                max_features[i] = axes_n_features[i]
            elif isinstance(max_features[i], numbers.Integral):
                check_scalar(
                    max_features[i],
                    name="max_features",
                    target_type=numbers.Integral,
                    min_val=1,
                    include_boundaries="left",
                )
            elif isinstance(max_features[i], float):
                check_scalar(
                    max_features[i],
                    name="max_features",
                    target_type=numbers.Real,
                    min_val=0.0,
                    max_val=1.0,
                    include_boundaries="right",
                )
                if max_features[i] > 0.0:
                    max_features[i] = max(1, int(self.max_features * axes_n_features[i]))
                else:
                    max_features = 0.

        self.max_features_ = max_features

        if self.max_leaf_nodes is not None:
            check_scalar(
                self.max_leaf_nodes,
                name="max_leaf_nodes",
                target_type=numbers.Integral,
                min_val=2,
            )
        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        check_scalar(
            self.min_impurity_decrease,
            name="min_impurity_decrease",
            target_type=numbers.Real,
            min_val=0.0,
        )

        for ax in range(2):
            if y.shape[ax] != X[ax].shape[0]:
                raise ValueError(
                    "Number of labels=%d does not match number of samples=%d"
                    "on axis %d"
                    % (y.shape[ax], X[ax].shape[0], ax)
                )

        if sample_weight is not None:
            pass  # FIXME
            #sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        if expanded_class_weight is not None:
            raise NotImplementedError
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * y.size
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * np.sum(sample_weight)

        return (
            max_depth,
            min_samples_leaf,
            min_samples_split,
            max_features,
            #self.max_features_ = max_features
            max_leaf_nodes,
            sample_weight,
            min_weight_leaf,
        )

    def _get_n_outputs(self, X, y):
        # TODO: Multi-output is not yet implemented for bipartite trees.
        return 1

    def _validate_X_predict(self, X, check_input):
        """Validate the training data on predict (probabilities)."""
        # FIXME: storing a whole matrix unnecessarily.
        if isinstance(X, (tuple, list)):  # FIXME: better criteria.
            X = row_cartesian_product(X)

        X = super()._validate_X_predict(X, check_input)
        return X


# =============================================================================
# Public estimators
# =============================================================================

class DecisionTreeRegressor2D(
     MultipartiteRegressorMixin, BaseDecisionTree2D, DecisionTreeRegressor
 ):
    """Adaptarion of sklearn's decision tree regressor to 2D input data.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"squared_error", "friedman_mse", "absolute_error", \
            "poisson"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

        .. deprecated:: 1.0
           `n_features_` is deprecated in 1.0 and will be removed in
           1.2. Use `n_features_in_` instead.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    DecisionTreeClassifier : A decision tree classifier.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> regressor = DecisionTreeRegressor(random_state=0)
    >>> cross_val_score(regressor, X, y, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([-0.39..., -0.46...,  0.02...,  0.06..., -0.50...,
           0.16...,  0.11..., -0.73..., -0.30..., -0.00...])
    """

    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,

        # 2D specific:
        ax_min_samples_leaf=1,
        ax_min_weight_fraction_leaf=None,
        ax_max_features=None,
        criterion_wrapper="squared_error",

        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,

            # 2D specific:
            ax_min_samples_leaf=ax_min_samples_leaf,
            ax_min_weight_fraction_leaf=ax_min_weight_fraction_leaf,
            ax_max_features=ax_max_features,
            criterion_wrapper=criterion_wrapper,

            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : list-like of {array-like, sparse matrix} of shapes (n_axis_samples,
            n_axis_features).
            The training input samples for each axis. Internally, it will be
            converted to ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_row_samples, n_col_samples)
            The target values (real numbers). Use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        sample_weight : array-like of shape (n_row_samples+n_col_samples,),
            default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.
            Row sample weights and column sample weights must be provided in
            one concatenated array.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self : DecisionTreeRegressor2D
            Fitted estimator.
        """

        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
        )
        return self


class ExtraTreeRegressor2D(DecisionTreeRegressor2D):
    """An extremely randomized tree regressor.
    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.
    Warning: Extra-trees should only be used within ensemble methods.
    Read more in the :ref:`User Guide <tree>`.
    Parameters
    ----------
    criterion : {"squared_error", "friedman_mse"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and "mae" for the
        mean absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.
        .. versionadded:: 0.24
            Poisson deviance criterion.
        .. deprecated:: 1.0
            Criterion "mse" was deprecated in v1.0 and will be removed in
            version 1.2. Use `criterion="squared_error"` which is equivalent.
        .. deprecated:: 1.0
            Criterion "mae" was deprecated in v1.0 and will be removed in
            version 1.2. Use `criterion="absolute_error"` which is equivalent.
    splitter : {"random", "best"}, default="random"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : int, float, {"auto", "sqrt", "log2"} or None, default=1.0
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        .. versionchanged:: 1.1
            The default of `max_features` changed from `"auto"` to `1.0`.
        .. deprecated:: 1.1
            The `"auto"` option was deprecated in 1.1 and will be removed
            in 1.3.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    random_state : int, RandomState instance or None, default=None
        Used to pick randomly the `max_features` used at each split.
        See :term:`Glossary <random_state>` for details.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
        .. versionadded:: 0.19
    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.
        .. versionadded:: 0.22
    Attributes
    ----------
    max_features_ : int
        The inferred value of max_features.
    n_features_ : int
        The number of features when ``fit`` is performed.
        .. deprecated:: 1.0
           `n_features_` is deprecated in 1.0 and will be removed in
           1.2. Use `n_features_in_` instead.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    feature_importances_ : ndarray of shape (n_features,)
        Return impurity-based feature importances (the higher, the more
        important the feature).
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.
    See Also
    --------
    ExtraTreeClassifier : An extremely randomized tree classifier.
    sklearn.ensemble.ExtraTreesClassifier : An extra-trees classifier.
    sklearn.ensemble.ExtraTreesRegressor : An extra-trees regressor.
    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.
    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import BaggingRegressor
    >>> from sklearn.tree import ExtraTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> extra_tree = ExtraTreeRegressor(random_state=0)
    >>> reg = BaggingRegressor(extra_tree, random_state=0).fit(
    ...     X_train, y_train)
    >>> reg.score(X_test, y_test)
    0.33...
    """
    def __init__(
        self,
        *,
        criterion="squared_error",

        # Only difference from DecisionTreeRegressor2D:
        splitter="random",

        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,

        # 2D parameters:
        ax_min_samples_leaf=1,
        ax_min_weight_fraction_leaf=None,
        ax_max_features=None,
        criterion_wrapper="squared_error",

        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,

            # 2D specific:
            ax_min_samples_leaf=ax_min_samples_leaf,
            ax_min_weight_fraction_leaf=ax_min_weight_fraction_leaf,
            ax_max_features=ax_max_features,
            criterion_wrapper=criterion_wrapper,

            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )


# TODO: docs
class BiclusteringTreeRegressor(DecisionTreeRegressor2D):
    """Implementation of Predictive Bi-Clustering Trees.

    Based on the original paper by Pliakos _et al._, 2018.
    DOI: 10.1007/s10994-018-5700-x

    By default, it used the GMO_{sa} approach, as described by the paper.
    if X are similarity kernels with 1 meaning equality, set
    `prediction_weights=1.` to employ pure GMO.

    Parameters
    ----------
    criterion : {"squared_error", "friedman_mse", "absolute_error", \
            "poisson"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22
    
    prediction_weights : {"uniform", "x", "raw"}, float, 1D-array or callable,
                         default="uniform"
        Determines how to compute the final predicted value. Initially, all
        predictions for each row and column instance from the training set that
        share the leaf node with the predicting sample are obtained.

        - "raw" instructs to return this vector, with a value for each training
          row and training column, and `np.nan` for instances not in the same
          leaf.
        - "uniform" returns the mean value of the leaf.

        Other options return the weighted average of the leaf values:

        - A 1D-array may be provided to specify training sample weights
          explicitly, with weights for training row samples followed by weights
          for training column samples (size=`sum(y_train.shape)`).
        - "x" instructs the estimator to consider x values as similarities to
          each row and column sample in the training set (row distances
          followed by column distances), so that the weights are `x`.
        - A callable, if provided, takes all the X being predicted and must
          return an array of weights for each predicting sample.
        - A `float` defines a similarity threshold from which a predicted
          value will be considered, also considering x as similarities to the
          training samples (Xs are kernel matrices). If no training instance is
          found to reach the theshold for a given predicting sample, the
          predicted value falls back to weight all training samples in the leaf
          as in the case where `prediction_weights`="x".

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

        .. deprecated:: 1.0
           `n_features_` is deprecated in 1.0 and will be removed in
           1.2. Use `n_features_in_` instead.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    DecisionTreeClassifier : A decision tree classifier.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> regressor = DecisionTreeRegressor(random_state=0)
    >>> cross_val_score(regressor, X, y, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([-0.39..., -0.46...,  0.02...,  0.06..., -0.50...,
           0.16...,  0.11..., -0.73..., -0.30..., -0.00...])
    """

    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,

        # 2D specific:
        ax_min_samples_leaf=1,
        ax_min_weight_fraction_leaf=None,
        ax_max_features=None,
        criterion_wrapper="local_multioutput",

        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,

        # PBCT-specific:
        prediction_weights="uniform",
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,

            # 2D specific:
            ax_min_samples_leaf=ax_min_samples_leaf,
            ax_min_weight_fraction_leaf=ax_min_weight_fraction_leaf,
            ax_max_features=ax_max_features,
            criterion_wrapper=criterion_wrapper,

            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )

        # PBCT-specific
        self.prediction_weights=prediction_weights
    
    def fit(self, X, y, sample_weight=None, check_input=True):
        # Validate prediction_weights.
        pw = self.prediction_weights

        if not isinstance(pw, (str, numbers.Number, np.ndarray, Callable)):
            raise TypeError("prediction_weights must be "
                            "a string, number, array, or callable object "
                            f"({pw} provided)")
        if isinstance(pw, str):
            if pw not in ("uniform", "x", "raw"):
                raise ValueError("Valid string values for prediction_weights "
                                 "are 'uniform', 'x' or 'raw'.")
        if (
            (isinstance(pw, str) and pw == "x") or
            isinstance(pw, (float, Callable))
        ):
            # TODO: properly check pairwise or set more_tags()
            for ax in range(len(X)):
                if X[ax].shape[0] != X[ax].shape[1]:
                    raise ValueError("X matrices must be square (pairwise) if "
                                     "prediction_weights is a float, callable "
                                     " or 'x'")
        elif isinstance(pw, np.ndarray):
            if pw.ndim != 1:
                raise ValueError("If an array, prediction_weights must be one-"
                                 f"dimensional.")
            n_outputs = self._get_n_outputs(X, y)
            if pw.size != n_outputs:
                raise ValueError("If an array, prediction_weights must be of "
                                 f"length self.n_outputs = {n_outputs}")

        super().fit(X, y, sample_weight, check_input)

    def _get_n_outputs(self, X, y):
        # The Criterion initially generates one output for each y row and
        # y column, with a bunch of np.nan to indicate samples not in the node.
        # These values are then processed by predict to yield a single output.
        # FIXME: can broke other code expecting n_outputs values from predict.
        return sum(y.shape)
    
    def predict(self, X, check_input=True):
        # TODO: use sample_weights?
        # TODO: normalize by n_rows and n_cols? The original paper does not.
        # TODO: use some sort of KNN object from sklearn
        # TODO: kernel parameters
        X = self._validate_X_predict(X, check_input)
        pred = super().predict(X, check_input=False)

        if isinstance(self.prediction_weights, str):
            if self.prediction_weights == "raw":
                return pred
            elif self.prediction_weights == "uniform":
                return np.nanmean(pred, axis=1)
            elif self.prediction_weights == "x":
                weights = X

        elif isinstance(self.prediction_weights, numbers.Number):
            # prediction_weights is a similarity threshold in this case
            weights = X >= self.prediction_weights
            zeroed_samples = ~weights.any(axis=1)
            # Fall back to use X as weights if no feature reaches the threshold
            weights[zeroed_samples] = X[zeroed_samples]

        elif isinstance(self.prediction_weights, np.ndarray):
            weights = self.prediction_weights

        elif isinstance(self.prediction_weights, Callable):
            weights = self.prediction_weights(X)
            n_samples = X.shape[0]
            if (
                weights.ndim != 2 or
                weights.shape[0] != n_samples or
                weights.shape[1] != self.n_outputs_
            ):
                raise ValueError(
                    "Callable prediction_weights must take a 2D array as input"
                    " and return another 2D array with the same number of rows"
                    " and `self.n_outputs_` columns (output shape was "
                    f"{weights.shape}, expected {(n_samples, self.n_outputs_)})"
                )

        weight_sum = np.sum(weights * ~np.isnan(pred), axis=-1, dtype=float)
        # Set predictions to zero if weight sum is zero
        weight_sum[weight_sum == 0] = np.inf
        return np.nansum(weights*pred, axis=1) / weight_sum


PBCT = BiclusteringTreeRegressor


# TODO: Create ABC, do not inherit from BiclusteringTreeRegressor
class BiclusteringExtraTreeRegressor(BiclusteringTreeRegressor):
    """An extremely randomized tree regressor.
    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.
    Warning: Extra-trees should only be used within ensemble methods.
    Read more in the :ref:`User Guide <tree>`.
    Parameters
    ----------
    criterion : {"squared_error", "friedman_mse"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and "mae" for the
        mean absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.
        .. versionadded:: 0.24
            Poisson deviance criterion.
        .. deprecated:: 1.0
            Criterion "mse" was deprecated in v1.0 and will be removed in
            version 1.2. Use `criterion="squared_error"` which is equivalent.
        .. deprecated:: 1.0
            Criterion "mae" was deprecated in v1.0 and will be removed in
            version 1.2. Use `criterion="absolute_error"` which is equivalent.
    splitter : {"random", "best"}, default="random"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for fractions.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : int, float, {"auto", "sqrt", "log2"} or None, default=1.0
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        .. versionchanged:: 1.1
            The default of `max_features` changed from `"auto"` to `1.0`.
        .. deprecated:: 1.1
            The `"auto"` option was deprecated in 1.1 and will be removed
            in 1.3.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    random_state : int, RandomState instance or None, default=None
        Used to pick randomly the `max_features` used at each split.
        See :term:`Glossary <random_state>` for details.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
        .. versionadded:: 0.19
    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.
        .. versionadded:: 0.22
    Attributes
    ----------
    max_features_ : int
        The inferred value of max_features.
    n_features_ : int
        The number of features when ``fit`` is performed.
        .. deprecated:: 1.0
           `n_features_` is deprecated in 1.0 and will be removed in
           1.2. Use `n_features_in_` instead.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    feature_importances_ : ndarray of shape (n_features,)
        Return impurity-based feature importances (the higher, the more
        important the feature).
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.
    See Also
    --------
    ExtraTreeClassifier : An extremely randomized tree classifier.
    sklearn.ensemble.ExtraTreesClassifier : An extra-trees classifier.
    sklearn.ensemble.ExtraTreesRegressor : An extra-trees regressor.
    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.
    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import BaggingRegressor
    >>> from sklearn.tree import ExtraTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> extra_tree = ExtraTreeRegressor(random_state=0)
    >>> reg = BaggingRegressor(extra_tree, random_state=0).fit(
    ...     X_train, y_train)
    >>> reg.score(X_test, y_test)
    0.33...
    """
    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="random",

        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,

        # 2D parameters:
        ax_min_samples_leaf=1,
        ax_min_weight_fraction_leaf=None,
        ax_max_features=None,
        criterion_wrapper="local_multioutput",

        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,

        # PBCT-specific:
        prediction_weights="uniform",
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,

            # 2D specific:
            ax_min_samples_leaf=ax_min_samples_leaf,
            ax_min_weight_fraction_leaf=ax_min_weight_fraction_leaf,
            ax_max_features=ax_max_features,
            criterion_wrapper=criterion_wrapper,

            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )

        # PBCT-specific
        self.prediction_weights=prediction_weights