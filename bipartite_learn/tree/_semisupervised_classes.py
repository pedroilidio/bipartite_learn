"""
This module gathers tree-based methods, including decision, regression and
randomized trees, adapted from sklearn for semi-supervised learning.
"""

# Author: Pedro Ilidio <pedrilidio@gmail.com>
# Adapted from scikit-learn.
#
# License: BSD 3 clause

from typing import Type
from copy import deepcopy
from abc import ABCMeta
from abc import abstractmethod
from math import ceil
from numbers import Integral, Real

import numpy as np
from scipy.sparse import issparse

from sklearn.base import is_classifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils._param_validation import (
    validate_params, Interval, StrOptions, Hidden,
)
from sklearn.metrics.pairwise import check_pairwise_arrays

from sklearn.tree._criterion import Criterion
from sklearn.tree._splitter import Splitter
from sklearn.tree._tree import (
    Tree, DepthFirstTreeBuilder, BestFirstTreeBuilder
)
from sklearn.tree import _tree
# Hypertree-specific:
from ._bipartite_splitter import BipartiteSplitter

# Semi-supervision-specific:
from sklearn.tree._classes import (
    BaseDecisionTree,
    DecisionTreeRegressor,
    DecisionTreeClassifier,
    DENSE_SPLITTERS,
    SPARSE_SPLITTERS,
)
from ._bipartite_classes import (
    BaseBipartiteDecisionTree,
    BipartiteDecisionTreeRegressor,
    BipartiteExtraTreeRegressor,
    _get_criterion_classes,
)
from . import (
    _semisupervised_splitter,
    _semisupervised_criterion,
    _unsupervised_criterion,
    _splitter_factory,
    _axis_criterion,
)


__all__ = [
    "DecisionTreeClassifierSS",
    "ExtraTreeClassifierSS",
    "DecisionTreeRegressorSS",
    "ExtraTreeRegressorSS",
    "BipartiteDecisionTreeRegressorSS",
]


# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

SINGLE_FEATURE_SPLITTERS = {
    "best": _semisupervised_splitter.BestSplitterSFSS,
    "random": _semisupervised_splitter.RandomSplitterSFSS,
}
SS_CRITERIA = {
    "default": _semisupervised_criterion.SSCompositeCriterion,
}

# Unuspervised criteria for numeric X (based on regression criteria).
U_NUMERIC_CRITERIA = {
    "squared_error": _unsupervised_criterion.UnsupervisedSquaredError,
    "friedman": _unsupervised_criterion.UnsupervisedFriedman,
    "pairwise_squared_error": _unsupervised_criterion.PairwiseSquaredError,
    "pairwise_friedman": _unsupervised_criterion.PairwiseFriedman,
    "mean_distance": _unsupervised_criterion.MeanDistance
}

# Unuspervised criteria for categoric X (based on classification criteria).
U_CATEGORIC_CRITERIA = {
    "gini": _unsupervised_criterion.UnsupervisedGini,
    "entropy": _unsupervised_criterion.UnsupervisedEntropy,
    "pairwise_gini": _unsupervised_criterion.PairwiseGini,
    "pairwise_entropy": _unsupervised_criterion.PairwiseEntropy,
}
UNSUPERVISED_CRITERIA = U_CATEGORIC_CRITERIA | U_NUMERIC_CRITERIA

PAIRWISE_CRITERION_OPTIONS = {
    "mean_distance",
    "pairwise_gini",
    "pairwise_entropy",
    "pairwise_squared_error",
    "pairwise_friedman",
}


def _encode_classes(y, class_weight=None):
    check_classification_targets(y)
    classes = []
    n_classes = []
    y_encoded = np.zeros(y.shape, dtype=int)

    for k in range(y.shape[1]):
        classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
        classes.append(classes_k)
        n_classes.append(classes_k.shape[0])

    if class_weight is None:
        expanded_class_weight = None
    else:
        expanded_class_weight = compute_sample_weight(class_weight, y)

    n_classes = np.array(n_classes, dtype=np.intp)

    return y_encoded, classes, n_classes, expanded_class_weight


@validate_params({
    "criterion": [str, Criterion, type(Criterion)]
})
def _is_categoric_criterion(criterion: str | Criterion | Type[Criterion]):
    # TODO: different base classes for categoric (classification) and
    # regression (numeric) unsupervised criteria.
    if isinstance(criterion, str):
        return criterion in U_CATEGORIC_CRITERIA
    if isinstance(criterion, type):
        return issubclass(criterion, tuple(U_CATEGORIC_CRITERIA.values())),
    if isinstance(criterion, Criterion):
        return issubclass(criterion, tuple(U_CATEGORIC_CRITERIA.values())),
    raise TypeError


def _is_pairwise_criterion(unsupervised_criterion):
    return (
        unsupervised_criterion in PAIRWISE_CRITERION_OPTIONS
        or isinstance(
            unsupervised_criterion,
            _unsupervised_criterion.PairwiseCriterion
        )
    )


def _validate_X_targets(
    *,
    X,
    X_targets,
    is_categoric,
    is_pairwise,
):
    if X_targets is None:
        # _X_targets will be the target matrix for unsupervised criteria,
        # and thus must be formatted like y usually is, double-valued and
        # C contiguous.
        X_targets = X.toarray() if issparse(X) else X

        if is_categoric:
            # TODO: class weights for X.
            X_targets, *_ = _encode_classes(X_targets)

        if (
            getattr(X_targets, "dtype", None) != DOUBLE
            or not X_targets.flags.contiguous
        ):
            X_targets = np.ascontiguousarray(X_targets, dtype=DOUBLE)

    elif (
        getattr(X_targets, "dtype", None) != DOUBLE
        or not X_targets.flags.contiguous
    ):
        raise TypeError(
            "If provided, _X_targets must be a C contiguous array of float64"
        )

    if is_pairwise:
        check_pairwise_arrays(X_targets, Y=None, precomputed=True)

    return X_targets


def _validate_bipartite_X_targets(
    *,
    X,
    X_targets,
    is_categoric_rows,
    is_categoric_cols,
    is_pairwise_rows,
    is_pairwise_cols,
):
    if X_targets is None:
        X_targets = [None, None]

    for axis, (is_categoric, is_pairwise) in enumerate((
        (is_categoric_rows, is_pairwise_rows),
        (is_categoric_cols, is_pairwise_cols),
    )):
        X_targets[axis] = _validate_X_targets(
            X=X[axis],
            X_targets=X_targets[axis],
            is_categoric=is_categoric,
            is_pairwise=is_pairwise,
        )

    return X_targets


# =============================================================================
# Semisupervised classes
# =============================================================================


class BaseDecisionTreeSS(BaseDecisionTree, metaclass=ABCMeta):
    """Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    _parameter_constraints: dict = {
        **BaseDecisionTree._parameter_constraints,
        "unsupervised_criterion": [
            StrOptions(set(UNSUPERVISED_CRITERIA.keys())),
            Hidden(Criterion),
        ],
        "supervision": [Interval(Real, 0.0, 1.0, closed="both")],
        "update_supervision": [callable, None],
        "ss_adapter": [
            None,
            Hidden(StrOptions(set(SS_CRITERIA.keys()))),
            Hidden(_semisupervised_criterion.SSCompositeCriterion),
        ],
        "preprocess_X_targets": [callable, None],
    }

    def _more_tags(self):
        # For cross-validation routines to split data correctly
        return {
            "pairwise": _is_pairwise_criterion(self.unsupervised_criterion)
        }

    @abstractmethod
    def __init__(
        self,
        *,
        supervision=0.5,
        unsupervised_criterion="squared_error",
        update_supervision=None,
        ss_adapter=None,
        preprocess_X_targets=None,
        _X_targets=None,
        **kwargs,
    ):
        self.supervision = supervision
        self.ss_adapter = ss_adapter
        self.unsupervised_criterion = unsupervised_criterion
        self.update_supervision = update_supervision
        self.preprocess_X_targets = preprocess_X_targets
        self._X_targets = _X_targets

        super().__init__(**kwargs)

    def _check_X_targets(self, X, _X_targets=None):
        if _X_targets is None and self.preprocess_X_targets is not None:
            X = self.preprocess_X_targets(X)

        return _validate_X_targets(
            X=X,
            X_targets=_X_targets,
            is_categoric=_is_categoric_criterion(
                self.unsupervised_criterion,
            ),
            is_pairwise=_is_pairwise_criterion(
                self.unsupervised_criterion,
            ),
        )

    # FIXME: Avoid copying the whole BaseDecisionTree.fit(). We currently do it
    # just so we can build the Splitter object our own way.
    def fit(self, X, y, sample_weight=None, check_input=True, _X_targets=None):
        self._validate_params()
        random_state = check_random_state(self.random_state)

        if check_input:
            # TODO: Since we cannot pass _X_targets to trees in an ensemble,
            # inside sklearn.ensemble._forest._parallel_build_trees(),
            # we add _X_targets as a parameter in self.__init__ and in the
            # forest.estimator_params list, that will be reused by the
            # individual trees without copying the array for each tree
            # and consuming a large amount of memory.  In forests, check_inpu
            # is set to false, in which case we do not override self._X_targets
            # below.
            self._X_targets = self._check_X_targets(X, _X_targets)

            # Need to validate separately here.
            # We can't pass multi_output=True because that would allow y to be
            # csr.
            check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
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
        n_samples, self.n_features_in_ = X.shape

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels=%d does not match number of samples=%d"
                % (len(y), n_samples)
            )

        is_classification = is_classifier(self)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            y, self.classes_, self.n_classes_, expanded_class_weight = \
                _encode_classes(y, self.class_weight)

        max_depth = np.iinfo(
            np.int32).max if self.max_depth is None else self.max_depth

        if isinstance(self.min_samples_leaf, Integral):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, Integral):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(
                    1, int(self.max_features * self.n_features_in_))
            else:
                max_features = 0

        self.max_features_ = max_features

        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * \
                np.sum(sample_weight)

        # Build tree
        splitter = self._make_splitter(
            X=X,
            X_targets=self._X_targets,
            min_samples_leaf=min_samples_leaf,
            min_weight_leaf=min_weight_leaf,
            random_state=random_state,
        )

        if is_classifier(self):
            self.tree_ = Tree(
                self.n_features_in_,
                self.n_classes_,
                self.n_outputs_,
            )
        else:
            self.tree_ = Tree(
                self.n_features_in_,
                # TODO: tree shouldn't need this in this case
                np.array([1] * self.n_outputs_, dtype=np.intp),
                self.n_outputs_,
            )

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
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

    def _check_criterion(
        self,
        *,
        n_samples,
        X_targets=None,
        n_outputs=None,
        n_features=None,
        supervision=None,
        ss_adapter=None,
        criterion=None,
        unsupervised_criterion=None,
        update_supervision=None,
    ):
        # FIXME: classification is not covered.
        n_outputs = n_outputs or self.n_outputs_
        n_features = n_features or self.n_features_in_
        supervision = supervision or self.supervision
        ss_adapter = ss_adapter or self.ss_adapter
        criterion = criterion or self.criterion
        update_supervision = update_supervision or self.update_supervision
        X_targets = X_targets if X_targets is not None else self._X_targets
        unsupervised_criterion = unsupervised_criterion or self.unsupervised_criterion

        if isinstance(ss_adapter, str):
            ss_adapter = SS_CRITERIA[ss_adapter]
        elif ss_adapter is not None:  # SSCompositeCriterion:
            # Make a deepcopy in case the splitter has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            return deepcopy(ss_adapter)

        if isinstance(criterion, str):
            if is_classifier(self):
                CRITERIA = U_CATEGORIC_CRITERIA
            else:
                CRITERIA = U_NUMERIC_CRITERIA
            criterion = CRITERIA[criterion]
        else:
            criterion = deepcopy(criterion)

        if isinstance(unsupervised_criterion, str):
            unsupervised_criterion = UNSUPERVISED_CRITERIA[unsupervised_criterion]
        else:
            unsupervised_criterion = deepcopy(unsupervised_criterion)

        final_criterion = _splitter_factory.make_semisupervised_criterion(
            ss_class=ss_adapter,
            supervision=supervision,
            supervised_criterion=criterion,
            unsupervised_criterion=unsupervised_criterion,
            n_outputs=n_outputs,
            n_features=n_features,
            n_samples=n_samples,
            update_supervision=update_supervision,
        )

        if X_targets is None:
            raise RuntimeError("_X_targets was not set.")

        final_criterion.set_X(X_targets)
        return final_criterion

    def _make_splitter(
        self,
        *,
        X,
        X_targets,
        min_samples_leaf,
        min_weight_leaf,
        random_state,
    ):
        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        if isinstance(self.splitter, Splitter):
            return deepcopy(self.splitter)
        else:
            criterion = self._check_criterion(
                X_targets=X_targets, n_samples=X.shape[0],
            )

            return SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
            )


# =============================================================================
# Public estimators
# =============================================================================


class DecisionTreeClassifierSS(BaseDecisionTreeSS, DecisionTreeClassifier):
    """A decision tree classifier (semi-supervised version).

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.

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
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None or 1.0, then `max_features=n_features`.

            .. deprecated:: 1.1
                The `"auto"` option was deprecated in 1.1 and will be removed
                in 1.3.

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

    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

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
    DecisionTreeRegressor : A decision tree regressor.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The :meth:`predict` method operates using the :func:`numpy.argmax`
    function on the outputs of :meth:`predict_proba`. This means that in
    case the highest predicted probabilities are tied, the classifier will
    predict the tied class with the lowest index in :term:`classes_`.

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
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    """

    _parameter_constraints: dict = {
        **BaseDecisionTreeSS._parameter_constraints,
        **DecisionTreeClassifier._parameter_constraints,
    }
    _parameter_constraints["criterion"] = [
        StrOptions(
            set(U_CATEGORIC_CRITERIA.keys())
            - PAIRWISE_CRITERION_OPTIONS,
        ),
        Hidden(_axis_criterion.BaseComposableCriterion),
    ]
    _parameter_constraints.pop("splitter")

    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        # Semi-supervised parameters:
        supervision=0.5,
        ss_adapter=None,
        unsupervised_criterion="squared_error",
        update_supervision=None,
        preprocess_X_targets=None,
        _X_targets=None,
    ):
        super().__init__(
            splitter=splitter,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            # Semi-supervised parameters:
            supervision=supervision,
            ss_adapter=ss_adapter,
            unsupervised_criterion=unsupervised_criterion,
            update_supervision=update_supervision,
            preprocess_X_targets=preprocess_X_targets,
            _X_targets=_X_targets,
        )


class DecisionTreeRegressorSS(BaseDecisionTreeSS, DecisionTreeRegressor):
    """A decision tree regressor (semi-supervised version).

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
        - If None or 1.0, then `max_features=n_features`.

        .. deprecated:: 1.1
            The `"auto"` option was deprecated in 1.1 and will be removed
            in 1.3.

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

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

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

    _parameter_constraints: dict = {
        **BaseDecisionTreeSS._parameter_constraints,
        **DecisionTreeRegressor._parameter_constraints,
    }
    _parameter_constraints["criterion"] = [
        StrOptions(
            set(U_NUMERIC_CRITERIA.keys())
            - PAIRWISE_CRITERION_OPTIONS,
        ),
        Hidden(_axis_criterion.BaseComposableCriterion),
    ]
    _parameter_constraints.pop("splitter")

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
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        # Semi-supervised parameters:
        supervision=0.5,
        ss_adapter=None,
        unsupervised_criterion="squared_error",
        update_supervision=None,
        preprocess_X_targets=None,
        _X_targets=None,
    ):
        super().__init__(
            splitter=splitter,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            # Semi-supervised parameters:
            supervision=supervision,
            ss_adapter=ss_adapter,
            unsupervised_criterion=unsupervised_criterion,
            update_supervision=update_supervision,
            preprocess_X_targets=preprocess_X_targets,
            _X_targets=_X_targets,
        )


class ExtraTreeClassifierSS(DecisionTreeClassifierSS):
    """An extremely randomized tree classifier (semi-supervised version).

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
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.

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

    max_features : int, float, {"auto", "sqrt", "log2"} or None, default="sqrt"
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

            .. versionchanged:: 1.1
                The default of `max_features` changed from `"auto"` to `"sqrt"`.

            .. deprecated:: 1.1
                The `"auto"` option was deprecated in 1.1 and will be removed
                in 1.3.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Used to pick randomly the `max_features` used at each split.
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

    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    max_features_ : int
        The inferred value of max_features.

    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    ExtraTreeRegressor : An extremely randomized tree regressor.
    sklearn.ensemble.ExtraTreesClassifier : An extra-trees classifier.
    sklearn.ensemble.ExtraTreesRegressor : An extra-trees regressor.
    sklearn.ensemble.RandomForestClassifier : A random forest classifier.
    sklearn.ensemble.RandomForestRegressor : A random forest regressor.
    sklearn.ensemble.RandomTreesEmbedding : An ensemble of
        totally random trees.

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
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import BaggingClassifier
    >>> from sklearn.tree import ExtraTreeClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...    X, y, random_state=0)
    >>> extra_tree = ExtraTreeClassifier(random_state=0)
    >>> cls = BaggingClassifier(extra_tree, random_state=0).fit(
    ...    X_train, y_train)
    >>> cls.score(X_test, y_test)
    0.8947...
    """

    def __init__(
        self,
        *,
        criterion="gini",
        splitter="random",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        # Semi-supervised parameters:
        supervision=0.5,
        ss_adapter=None,
        unsupervised_criterion="squared_error",
        update_supervision=None,
        preprocess_X_targets=None,
        _X_targets=None,
    ):
        super().__init__(
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
            # Semi-supervised parameters:
            supervision=supervision,
            ss_adapter=ss_adapter,
            criterion=criterion,
            unsupervised_criterion=unsupervised_criterion,
            update_supervision=update_supervision,
            preprocess_X_targets=preprocess_X_targets,
            _X_targets=_X_targets,
        )


class ExtraTreeRegressorSS(DecisionTreeRegressorSS):
    """An extremely randomized tree regressor (semi-supervised version).

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

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

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
        max_features=1.0,
        random_state=None,
        min_impurity_decrease=0.0,
        max_leaf_nodes=None,
        ccp_alpha=0.0,
        # Semi-supervised parameters:
        supervision=0.5,
        ss_adapter=None,
        unsupervised_criterion="squared_error",
        update_supervision=None,
        preprocess_X_targets=None,
        _X_targets=None,
    ):
        super().__init__(
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
            # Semi-supervised parameters:
            supervision=supervision,
            ss_adapter=ss_adapter,
            criterion=criterion,
            unsupervised_criterion=unsupervised_criterion,
            update_supervision=update_supervision,
            preprocess_X_targets=preprocess_X_targets,
            _X_targets=_X_targets,
        )


# =============================================================================
# Bipartite semi-supervised trees
# =============================================================================


class BaseBipartiteDecisionTreeSS(
    BaseBipartiteDecisionTree,
    metaclass=ABCMeta,
):
    _parameter_constraints: dict = {
        **BaseBipartiteDecisionTree._parameter_constraints,
        **BaseDecisionTreeSS._parameter_constraints,
        "unsupervised_criterion_rows": [
            StrOptions(set(UNSUPERVISED_CRITERIA.keys())),
            Hidden(Criterion),
        ],
        "unsupervised_criterion_cols": [
            StrOptions(set(UNSUPERVISED_CRITERIA.keys())),
            Hidden(Criterion),
        ],
        "preprocess_X_targets": [callable, None],
    }
    _parameter_constraints.pop("unsupervised_criterion")

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
        # Bipartite parameters:
        min_rows_split=1,  # Not 2, to still allow splitting on the other axis
        min_cols_split=1,
        min_rows_leaf=1,
        min_cols_leaf=1,
        min_row_weight_fraction_leaf=0.0,
        min_col_weight_fraction_leaf=0.0,
        max_row_features=None,
        max_col_features=None,
        bipartite_adapter="gmosa",
        prediction_weights=None,
        # Semi-supervised parameters:
        supervision=0.5,
        ss_adapter=None,
        unsupervised_criterion_rows="squared_error",
        unsupervised_criterion_cols="squared_error",
        update_supervision=None,
        axis_decision_only=False,
        preprocess_X_targets=None,
    ):
        self.supervision = supervision
        self.ss_adapter = ss_adapter
        self.unsupervised_criterion_rows = unsupervised_criterion_rows
        self.unsupervised_criterion_cols = unsupervised_criterion_cols
        self.update_supervision = update_supervision
        self.axis_decision_only = axis_decision_only
        self.preprocess_X_targets = preprocess_X_targets

        BaseBipartiteDecisionTree.__init__(
            self,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            min_rows_split=min_rows_split,
            min_cols_split=min_cols_split,
            min_rows_leaf=min_rows_leaf,
            min_cols_leaf=min_cols_leaf,
            min_row_weight_fraction_leaf=min_row_weight_fraction_leaf,
            min_col_weight_fraction_leaf=min_col_weight_fraction_leaf,
            max_row_features=max_row_features,
            max_col_features=max_col_features,
            bipartite_adapter=bipartite_adapter,
            prediction_weights=prediction_weights,
        )

    def _more_tags(self):
        # For cross-validation routines to split data correctly
        tags = {
            "pairwise_rows": (
                _is_pairwise_criterion(self.unsupervised_criterion_rows)
                or self.prediction_weights is not None
            ),
            "pairwise_cols": (
                _is_pairwise_criterion(self.unsupervised_criterion_cols)
                or self.prediction_weights is not None
            )
        }
        tags["pairwise"] = tags["pairwise_rows"] and tags["pairwise_cols"]
        return tags

    def fit(self, X, y, sample_weight=None, check_input=True, _X_targets=None):
        if check_input:
            _X_targets = self._check_X_targets(X, _X_targets)

        self._X_targets = _X_targets  # TODO: Keep it?

        return super().fit(X, y, sample_weight, check_input)

    def _check_X_targets(self, X, _X_targets=None):
        if _X_targets is None and self.preprocess_X_targets is not None:
            X = [self.preprocess_X_targets(Xi) for Xi in X]

        return _validate_bipartite_X_targets(
            X=X,
            X_targets=_X_targets,
            is_categoric_rows=_is_categoric_criterion(
                self.unsupervised_criterion_rows,
            ),
            is_categoric_cols=_is_categoric_criterion(
                self.unsupervised_criterion_cols,
            ),
            is_pairwise_rows=_is_pairwise_criterion(
                self.unsupervised_criterion_rows,
            ),
            is_pairwise_cols=_is_pairwise_criterion(
                self.unsupervised_criterion_cols,
            ),
        )

    def _make_splitter(
        self,
        *,
        X,
        n_outputs,
        n_samples,
        n_classes,
        min_samples_leaf,
        min_weight_leaf,
        ax_max_features,
        ax_min_samples_leaf,
        ax_min_weight_leaf,
        random_state,
    ):
        if isinstance(self.splitter, BipartiteSplitter):
            # Make a deepcopy in case the splitter has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            return deepcopy(self.splitter)

        # NOTE: It is possible to use diferent ss_adapter for rows and columns,
        #       but the user needs to pass an already built BipartiteSplitter instance
        #       if they want to do so.
        if isinstance(self.ss_adapter, _semisupervised_criterion.SSCompositeCriterion):
            ss_adapter = deepcopy(self.ss_adapter)
        elif isinstance(self.ss_adapter, str):
            ss_adapter = SS_CRITERIA[self.ss_adapter]
        elif self.ss_adapter is None:
            ss_adapter = None
        else:
            raise ValueError  # validate_params will not allow this to run

        if isinstance(self.unsupervised_criterion_rows, str):
            unsup_criterion_rows = \
                UNSUPERVISED_CRITERIA[self.unsupervised_criterion_rows]
        else:
            unsup_criterion_rows = deepcopy(self.unsupervised_criterion_rows)

        if isinstance(self.unsupervised_criterion_cols, str):
            unsup_criterion_cols = \
                UNSUPERVISED_CRITERIA[self.unsupervised_criterion_cols]
        else:
            unsup_criterion_cols = deepcopy(self.unsupervised_criterion_cols)

        bipartite_adapter = self.bipartite_adapter
        criterion = self.criterion

        if isinstance(criterion, str) and isinstance(bipartite_adapter, str):
            bipartite_adapter, criterion = _get_criterion_classes(
                adapter=bipartite_adapter,
                criterion=criterion,
                is_classification=is_classifier(self),
            )
        elif isinstance(criterion, str) or isinstance(bipartite_adapter, str):
            raise ValueError(
                "Either both or none of criterion and bipartite_adapter params"
                " must be strings."
            )
        else:  # Criterion instance or subclass
            criterion = deepcopy(criterion)

        splitter = self.splitter

        # User is able to specify a splitter for each axis
        if not isinstance(splitter, (tuple, list)):
            splitter = [splitter, splitter]
        for ax in range(2):
            if isinstance(splitter[ax], str):
                if issparse(X[ax]):
                    splitter[ax] = SPARSE_SPLITTERS[splitter[ax]]
                else:
                    splitter[ax] = DENSE_SPLITTERS[splitter[ax]]
            else:  # is a Splitter instance
                splitter[ax] = deepcopy(splitter[ax])

        splitter = _splitter_factory.make_bipartite_ss_splitter(
            splitters=splitter,
            supervised_criteria=criterion,
            unsupervised_criteria=[
                unsup_criterion_rows,
                unsup_criterion_cols,
            ],
            ss_criteria=ss_adapter,
            supervision=self.supervision,
            update_supervision=self.update_supervision,
            n_samples=n_samples,
            n_features=[
                self.n_row_features_in_,
                self.n_col_features_in_,
            ],
            n_classes=n_classes,
            n_outputs=n_outputs,
            max_features=ax_max_features,
            min_samples_leaf=min_samples_leaf,
            min_weight_leaf=min_weight_leaf,
            ax_min_samples_leaf=ax_min_samples_leaf,
            ax_min_weight_leaf=ax_min_weight_leaf,
            random_state=random_state,
            bipartite_criterion_class=bipartite_adapter,
            axis_decision_only=self.axis_decision_only,
        )

        if self._X_targets is None:
            raise RuntimeError("_X_targets was not set.")

        splitter.bipartite_criterion.set_X(
            self._X_targets[0],
            self._X_targets[1],
        )
        return splitter


class BipartiteDecisionTreeRegressorSS(
    BaseBipartiteDecisionTreeSS,
    BipartiteDecisionTreeRegressor,
):

    _parameter_constraints: dict = {
        **BaseBipartiteDecisionTreeSS._parameter_constraints,
        **BipartiteDecisionTreeRegressor._parameter_constraints,
    }
    _parameter_constraints.pop("splitter")

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
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        # Bipartite parameters:
        min_rows_split=1,  # Not 2, to still allow splitting on the other axis
        min_cols_split=1,
        min_rows_leaf=1,
        min_cols_leaf=1,
        min_row_weight_fraction_leaf=0.0,
        min_col_weight_fraction_leaf=0.0,
        max_row_features=None,
        max_col_features=None,
        bipartite_adapter="gmosa",
        prediction_weights=None,
        # Semi-supervised parameters:
        supervision=0.5,
        ss_adapter=None,
        unsupervised_criterion_rows="squared_error",
        unsupervised_criterion_cols="squared_error",
        update_supervision=None,
        axis_decision_only=False,
        preprocess_X_targets=None,
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
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            # Bipartite parameters:
            min_rows_split=min_rows_split,
            min_cols_split=min_cols_split,
            min_rows_leaf=min_rows_leaf,
            min_cols_leaf=min_cols_leaf,
            min_row_weight_fraction_leaf=min_row_weight_fraction_leaf,
            min_col_weight_fraction_leaf=min_col_weight_fraction_leaf,
            max_row_features=max_row_features,
            max_col_features=max_col_features,
            bipartite_adapter=bipartite_adapter,
            prediction_weights=prediction_weights,
            # Semi-supervised parameters:
            supervision=supervision,
            ss_adapter=ss_adapter,
            unsupervised_criterion_rows=unsupervised_criterion_rows,
            unsupervised_criterion_cols=unsupervised_criterion_cols,
            update_supervision=update_supervision,
            axis_decision_only=axis_decision_only,
            preprocess_X_targets=preprocess_X_targets,
        )


class BipartiteExtraTreeRegressorSS(
    BaseBipartiteDecisionTreeSS,
    BipartiteExtraTreeRegressor,
):

    _parameter_constraints: dict = {
        **BaseBipartiteDecisionTreeSS._parameter_constraints,
        **BipartiteExtraTreeRegressor._parameter_constraints,
    }
    _parameter_constraints.pop("splitter")

    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="random",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        # Bipartite parameters:
        min_rows_split=1,  # Not 2, to still allow splitting on the other axis
        min_cols_split=1,
        min_rows_leaf=1,
        min_cols_leaf=1,
        min_row_weight_fraction_leaf=0.0,
        min_col_weight_fraction_leaf=0.0,
        max_row_features=None,
        max_col_features=None,
        bipartite_adapter="gmosa",
        prediction_weights=None,
        # Semi-supervised parameters:
        supervision=0.5,
        ss_adapter=None,
        unsupervised_criterion_rows="squared_error",
        unsupervised_criterion_cols="squared_error",
        update_supervision=None,
        axis_decision_only=False,
        preprocess_X_targets=None,
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
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            # Bipartite parameters:
            min_rows_split=min_rows_split,
            min_cols_split=min_cols_split,
            min_rows_leaf=min_rows_leaf,
            min_cols_leaf=min_cols_leaf,
            min_row_weight_fraction_leaf=min_row_weight_fraction_leaf,
            min_col_weight_fraction_leaf=min_col_weight_fraction_leaf,
            max_row_features=max_row_features,
            max_col_features=max_col_features,
            bipartite_adapter=bipartite_adapter,
            prediction_weights=prediction_weights,
            # Semi-supervised parameters:
            supervision=supervision,
            ss_adapter=ss_adapter,
            unsupervised_criterion_rows=unsupervised_criterion_rows,
            unsupervised_criterion_cols=unsupervised_criterion_cols,
            update_supervision=update_supervision,
            axis_decision_only=axis_decision_only,
            preprocess_X_targets=preprocess_X_targets,
        )
