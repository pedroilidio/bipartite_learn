"""
This module gathers tree-based methods, including decision, regression and
randomized trees, adapted from sklearn for 2D training data.
"""
# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# 2D adaptation: Pedro Ilidio <pedrilidio@gmail.com>
#
# License: BSD 3 clause

import numbers
import warnings
import copy
from abc import ABCMeta
from abc import abstractmethod
from math import ceil

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
from sklearn.tree import _tree, _splitter, _criterion, DecisionTreeRegressor

# ND new:
from itertools import product
from sklearn.tree._classes import BaseDecisionTree
from ._nd_tree import DepthFirstTreeBuilder2D
from ._nd_criterion import MSE_Wrapper2D
from ._nd_splitter import Splitter2D, make_2d_splitter


__all__ = [
    "DecisionTreeRegressor2D",
    "PBCT",  # Alias to DecisionTreeRegressor2D.
]


# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = {}
CRITERIA_REG = {
    "squared_error": _criterion.MSE,
}

DENSE_SPLITTERS = {
    "best": _splitter.BestSplitter,
    "random": _splitter.RandomSplitter,
}

SPARSE_SPLITTERS = {}

# =============================================================================
# Base ND decision tree
# =============================================================================


class BaseDecisionTree2D(BaseDecisionTree, metaclass=ABCMeta):
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
        # TODO: ax_min_samples_split,
        ax_min_samples_leaf=1,  # New!
        ax_min_weight_fraction_leaf=None,  # New!
        ax_max_features=None,  # New!
        max_leaf_nodes,
        random_state,
        min_impurity_decrease,
        class_weight=None,
        ccp_alpha=0.0,
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features

        self.ax_min_samples_leaf = ax_min_samples_leaf
        self.ax_min_weight_fraction_leaf = ax_min_weight_fraction_leaf
        self.ax_max_features = ax_max_features

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


        # FIXME: enable check_input.
        # It currently tests the number of outputs and fail.
        if False and check_input:
            # We can't pass multi_ouput=True because that would allow y to be
            # csr.
            check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            y = self._validate_data(y, **check_y_params)

            for ax in range(len(X)):
                # FIXME: it will test the # of outputs and fail.
                X[ax] = self._validate_data(X[ax], **check_X_params)
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
        n_attrs = [Xax.shape[1] for Xax in X]
        n_samples, self.n_features_in_ = y.size, sum(n_attrs)
        is_classification = is_classifier(self)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        # self.n_outputs_ = y.shape[-1]  # TODO: implement.
        self.n_outputs_ = 1

        if is_classification:
            raise NotImplementedError("Let's not talk about classification.")
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


        (max_depth,
        min_samples_leaf,
        min_samples_split,
        max_features,
        # self.max_features_ = max_features
        max_leaf_nodes,
        sample_weight,
        min_weight_leaf) = self._check_parameters(
            X, y, sample_weight, expanded_class_weight)

        # TODO: move to _check_parameters.
        if self.ax_max_features is None:
            ax_max_features = n_attrs
        else:
            ax_max_features = self.ax_max_features

        ax_min_samples_leaf = self.ax_min_samples_leaf  # Defaults to 1.

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
            split_indices = np.cumsum(y.shape)
            ax_sample_weight = np.split(sample_weight, split_indices)[:-1]
            weighted_n_samples = np.prod([
                np.sum(sw) for sw in _ax_sample_weight])
            ax_min_weight_leaf = [
                mw * weighted_n_samples for mw in
                self.ax_min_weight_fraction_leaf
            ]

        # Build tree
        criterion = self.criterion
        if isinstance(criterion, str):
            if is_classification:
                criterion = CRITERIA_CLF[self.criterion]
            else:
                criterion = CRITERIA_REG[self.criterion]

        # NOTE: make_2d_splitter takes charge of that.
        # Make a deepcopy in case the criterion has mutable attributes that
        # might be shared and modified concurrently during parallel fitting
        # criterion = copy.deepcopy(criterion)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(splitter, Splitter2D):
            if isinstance(splitter, str):
                splitter = SPLITTERS[self.splitter]

            splitter = make_2d_splitter(
                splitter_class=splitter,
                criterion_class=criterion,
                n_samples=y.shape,
                n_outputs=self.n_outputs_,
                # TODO: check ax_* parameters.
                max_features=ax_max_features,
                min_samples_leaf=min_samples_leaf,
                min_weight_leaf=min_weight_leaf,
                ax_min_samples_leaf=ax_min_samples_leaf,
                ax_min_weight_leaf=ax_min_weight_leaf,
                random_state=random_state,
            )


        if is_classifier(self):
            self.tree_ = Tree(self.n_features_in_, self.n_classes_, self.n_outputs_)
        else:
            self.tree_ = Tree(
                self.n_features_in_,
                # TODO: tree shouldn't need this in this case
                np.array([1] * self.n_outputs_, dtype=np.intp),
                self.n_outputs_,
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

    def _check_parameters(self, X, y, sample_weight, expanded_class_weight):
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

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classification:
                    max_features = [max(1, int(np.sqrt(n)))
                                    for n in axes_n_features]
                else:
                    max_features = self.n_features_in_
            elif self.max_features == "sqrt":
                max_features = [max(1, int(np.sqrt(n))) for n in axes_n_features]
            elif self.max_features == "log2":
                max_features = [max(1, int(np.log2(n))) for n in axes_n_features]
            else:
                raise ValueError(
                    "Invalid value for max_features. "
                    "Allowed string values are 'auto', "
                    "'sqrt' or 'log2'."
                )
        elif self.max_features is None:
            max_features = axes_n_features
        elif isinstance(self.max_features, numbers.Integral):
            for ax in range(2):
                check_scalar(
                    self.max_features[ax],
                    name="max_features",
                    target_type=numbers.Integral,
                    min_val=1,
                    include_boundaries="left",
                )
            max_features = [self.max_features, self.max_features]
        elif isinstance(self.max_features[0], float):
            for ax in range(2):
                check_scalar(
                    self.max_features[ax],
                    name="max_features",
                    target_type=numbers.Real,
                    min_val=0.0,
                    max_val=1.0,
                    include_boundaries="right",
                )
            if self.max_features[0] > 0.0 or self.max_features[1] > 0.0:
                max_features = [max(1, int(self.max_features * n))
                                for n in axes_n_features]
            else:
                max_features = [0, 0]

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
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

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

    def _validate_X_predict(self, X, check_input):
        """Validate the training data on predict (probabilities)."""
        # FIXME: storing a whole matrix unnecessarily.
        if type(X) in (tuple, list) and len(X) == 2:  # FIXME: better criteria.
            X = np.array([np.hstack(x) for x in product(*X)])
        # return super()._validate_X_predict(X, check_input)  # FIXMEJ
        return X

#     # FIXME: reshape after?
#     def predict(self, X, check_input=True):
#         # Identify if each axis instances are provided separately.
#         # FIXME: better criteria.
#         axes_format = type(X) in (tuple, list) and len(X) == 2
# 
#         if axes_format:
#             X = np.fromiter(
#                 (np.hstack(x) for x in itertools.product(*X)),
#                 dtype=X[0].dtype)
#             original_shape = (len(Xax) for Xax in X)
# 
#         pred = super().predict(X, check_input)
# 
#         if axes_format:
#             pred = pred.reshape(*original_shape)
#         return pred


# =============================================================================
# Public estimators
# =============================================================================

# TODO
# class DecisionTreeClassifier2D(ClassifierMixin, BaseDecisionTree2D):
#     pass

class DecisionTreeRegressor2D(RegressorMixin, BaseDecisionTree2D):
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
        ax_min_samples_leaf=1,  # New!
        ax_min_weight_fraction_leaf=None,  # New!
        ax_max_features=None,  # New!
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
            ax_min_samples_leaf=ax_min_samples_leaf,
            ax_min_weight_fraction_leaf=ax_min_weight_fraction_leaf,
            ax_max_features=ax_max_features,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )

    # FIXME: fix check_input to set default to True.
    def fit(self, X, y, sample_weight=None, check_input=False):
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

    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray of shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray of shape (n_target_features)
            The set of target features for which the partial dependence
            should be evaluated.

        Returns
        -------
        averaged_predictions : ndarray of shape (n_samples,)
            The value of the partial dependence function on each grid point.
        """
        return DecisionTreeRegressor._compute_partial_dependence_recursion(
            self, grid, target_features)


PBCT = DecisionTreeRegressor2D  # Alias.

# TODO
# class ExtraTreeClassifier2D(DecisionTreeClassifier2D):
#     pass
# 
# class ExtraTreeRegressor2D(DecisionTreeRegressor2D):
#     pass
