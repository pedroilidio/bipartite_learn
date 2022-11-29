"""
This module gathers tree-based methods, including decision, regression and
randomized trees, adapted from sklearn for bipartite training data.
"""

# Author: Pedro Ilidio <pedrilidio@gmail.com>
# Adapted from scikit-learn.
#
# License: BSD 3 clause

from numbers import Integral, Real
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
from sklearn.utils.validation import (
    _check_sample_weight,
    _is_arraylike_not_scalar,
    check_is_fitted,
)
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets

from sklearn.utils._param_validation import Interval, StrOptions, Hidden

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
from ._nd_splitter import Splitter2D
from ..melter import row_cartesian_product
from ..utils import check_similarity_matrix, _X_is_multipartite
from ._axis_criterion import AxisMSE
from ._splitter_factory import make_2d_splitter


__all__ = [
    "BipartiteDecisionTreeRegressor",
    "BipartiteExtraTreeRegressor",
]


# =============================================================================
# Types and constants
# =============================================================================

BIPARTITE_CRITERIA = {
    "global_single_output": MSE_Wrapper2D,
    "local_multioutput": PBCTCriterionWrapper,
}

AXIS_CRITERIA_REG = {
    "squared_error": AxisMSE,
}

SPARSE_SPLITTERS = {}


def _normalize_weights(weights, pred):
    not_nan = ~np.isnan(pred)
    valid_weights = not_nan * weights

    # If a row contains a 1-valued weight, set the others weights of
    # that row to 0 (simmilarity == 1 means identity).
    is_known = (valid_weights == 1.).any(axis=1).reshape(-1)
    known_weights = valid_weights[is_known]
    known_weights[known_weights != 1.] = 0.

    # Fall back to uniform weights if weight sum is 0
    all_zero = (valid_weights == 0.).all(axis=1).reshape(-1)
    valid_weights[all_zero] = not_nan[all_zero]

    valid_weights /= valid_weights.sum(axis=1, keepdims=True)

    return valid_weights


# =============================================================================
# Base bipartite decision tree
# =============================================================================


class BaseBipartiteDecisionTree(BaseMultipartiteEstimator, BaseDecisionTree,
                                metaclass=ABCMeta):
    """Base class for bipartite decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    _parameter_constraints: dict = BaseDecisionTree._parameter_constraints | {
        "splitter": [StrOptions({"best", "random"}), Hidden(Splitter2D)],
        "min_rows_split": [
            # min value is not 2 to still allow split on the other axis.
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="right"),
        ],
        "min_cols_split": [
            # min value is not 2 to still allow split on the other axis.
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="right"),
        ],
        "min_rows_leaf": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="neither"),
        ],
        "min_cols_leaf": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="neither"),
        ],
        "min_row_weight_fraction_leaf": [
            Interval(Real, 0.0, 0.5, closed="both"),
        ],
        "min_col_weight_fraction_leaf": [
            Interval(Real, 0.0, 0.5, closed="both")
        ],
        "max_row_features": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="right"),
            StrOptions({"sqrt", "log2"}),
            None,
        ],
        "max_col_features": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="right"),
            StrOptions({"sqrt", "log2"}),
            None,
        ],
        "bipartite_adapter": [
            StrOptions({"global_single_output", "local_multioutput"}),
        ],
        "prediction_weights": [
            "array-like",
            StrOptions({"precomputed", "leaf_uniform", "uniform", "raw"}),
            callable,
            None,
        ],
    }

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
        min_rows_split=1,  # Not 2, to still allow split on the other axis
        min_cols_split=1,
        min_rows_leaf=1,
        min_cols_leaf=1,
        min_row_weight_fraction_leaf=0.0,
        min_col_weight_fraction_leaf=0.0,
        max_row_features=None,
        max_col_features=None,
        bipartite_adapter="global_single_output",
        prediction_weights=None,
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.min_rows_split = min_rows_split
        self.min_cols_split = min_cols_split
        self.min_rows_leaf = min_rows_leaf
        self.min_cols_leaf = min_cols_leaf
        self.min_row_weight_fraction_leaf = min_row_weight_fraction_leaf
        self.min_col_weight_fraction_leaf = min_col_weight_fraction_leaf
        self.max_row_features = max_row_features
        self.max_col_features = max_col_features
        self.bipartite_adapter = bipartite_adapter
        self.prediction_weights = prediction_weights

    def fit(self, X, y, row_weight=None, col_weight=None, check_input=True):
        self._validate_params()
        random_state = check_random_state(self.random_state)

        if check_input:
            # Need to validate separately here.
            # We can't pass multi_output=True because that would allow y to be
            # csr.
            check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )
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

        if y.shape[0] != len(X[0]) or y.shape[1] != len(X[1]):
            raise ValueError(
                f"Interaction matrix shape {y.shape=} does not match number "
                f"of samples {(len(X[0]),len(X[1]))=}"
            )

        # Determine output settings
        # n_samples, self.n_features_in_ = X.shape
        n_samples = y.size
        n_rows, n_cols = y.shape
        self._n_rows_fit = n_rows
        self.n_row_features_in_ = X[0].shape[1]
        self.n_col_features_in_ = X[1].shape[1]

        is_classification = is_classifier(self)

        y = np.atleast_1d(y)
        expanded_class_weight = None
        expanded_class_weight_rows = None
        expanded_class_weight_cols = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            # Columns do not represent different outputs anymore
            # for k in range(self.n_outputs_):
            classes, y = np.unique(y, return_inverse=True)
            self.classes_.append(classes)
            self.n_classes_.append(classes.shape[0])

            if self.class_weight is not None:
                expanded_class_weight_rows = compute_sample_weight(
                    self.class_weight, y_original
                )
                expanded_class_weight_cols = compute_sample_weight(
                    self.class_weight, y_original.T
                )

            self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        max_depth = np.iinfo(
            np.int32).max if self.max_depth is None else self.max_depth

        if isinstance(self.min_samples_leaf, Integral):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        # Bipartite validation
        if isinstance(self.min_rows_leaf, Integral):
            min_rows_leaf = self.min_rows_leaf
        else:  # float
            min_rows_leaf = int(ceil(self.min_rows_leaf * n_rows))
        if isinstance(self.min_cols_leaf, Integral):
            min_cols_leaf = self.min_cols_leaf
        else:  # float
            min_cols_leaf = int(ceil(self.min_cols_leaf * n_cols))

        if isinstance(self.min_samples_split, Integral):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        # Bipartite validation
        if isinstance(self.min_rows_split, Integral):
            min_rows_split = self.min_rows_split
        else:  # float
            min_rows_split = int(ceil(self.min_rows_split * n_rows))
        if isinstance(self.min_cols_split, Integral):
            min_cols_split = self.min_cols_split
        else:  # float
            min_cols_split = int(ceil(self.min_cols_split * n_cols))

        # TODO: Not implemented. Would have to dynamically change each
        #       splitter's max_features so that the total features selected
        #       considering both axes is constant.
        if self.max_features is not None:
            raise NotImplementedError(
                "max_features!=None is not implemented.Please set"
                "'max_row_features' and 'max_col_features' instead."
            )
        max_features = self.n_features_in_

        # if isinstance(self.max_features, str):
        #     if self.max_features == "sqrt":
        #         max_features = max(1, int(np.sqrt(self.n_features_in_)))
        #     elif self.max_features == "log2":
        #         max_features = max(1, int(np.log2(self.n_features_in_)))
        # elif self.max_features is None:
        #     max_features = self.n_features_in_
        # elif isinstance(self.max_features, Integral):
        #     max_features = self.max_features
        # else:  # float
        #     if self.max_features > 0.0:
        #         max_features = \
        #             max(1, int(self.max_features * self.n_features_in_))
        #     else:
        #         max_features = 0

        # Bipartite validation
        if isinstance(self.max_row_features, str):
            if self.max_row_features == "sqrt":
                max_row_features = max(
                    1, int(np.sqrt(self.n_row_features_in_)))
            elif self.max_row_features == "log2":
                max_row_features = max(
                    1, int(np.log2(self.n_row_features_in_)))
        elif self.max_row_features is None:
            max_row_features = self.n_row_features_in_
        elif isinstance(self.max_row_features, Integral):
            max_row_features = self.max_row_features
        else:  # float
            if self.max_row_features > 0.0:
                max_row_features = \
                    max(1, int(self.max_row_features * self.n_row_features_in_))
            else:
                max_row_features = 0

        if isinstance(self.max_col_features, str):
            if self.max_col_features == "sqrt":
                max_col_features = max(
                    1, int(np.sqrt(self.n_col_features_in_)))
            elif self.max_col_features == "log2":
                max_col_features = max(
                    1, int(np.log2(self.n_col_features_in_)))
        elif self.max_col_features is None:
            max_col_features = self.n_col_features_in_
        elif isinstance(self.max_col_features, Integral):
            max_col_features = self.max_col_features
        else:  # float
            if self.max_col_features > 0.0:
                max_col_features = \
                    max(1, int(self.max_col_features * self.n_col_features_in_))
            else:
                max_col_features = 0

        self.max_features_ = max_features
        self.max_row_features_ = max_row_features
        self.max_col_features_ = max_col_features

        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        if row_weight is not None:
            row_weight = _check_sample_weight(row_weight, X[0], DOUBLE)
        if col_weight is not None:
            col_weight = _check_sample_weight(col_weight, X[1], DOUBLE)

        if expanded_class_weight_rows is not None:
            if row_weight is not None:
                row_weight = row_weight * expanded_class_weight
            else:
                row_weight = expanded_class_weight_rows

        if expanded_class_weight_cols is not None:
            if col_weight is not None:
                col_weight = col_weight * expanded_class_weight
            else:
                col_weight = expanded_class_weight_cols

        # Set min_weight_leaf from min_weight_fraction_leaf
        row_weight_sum = n_rows if row_weight is None else np.sum(row_weight)
        col_weight_sum = n_cols if col_weight is None else np.sum(col_weight)

        min_row_weight_leaf = self.min_row_weight_fraction_leaf * row_weight_sum
        min_col_weight_leaf = self.min_col_weight_fraction_leaf * col_weight_sum

        min_weight_leaf = \
            self.min_weight_fraction_leaf * (row_weight_sum * col_weight_sum)

        # self.n_outputs_ = y.shape[-1]  # TODO: implement multi-output (3D y).
        self.n_outputs_ = 1

        if self.bipartite_adapter == "local_multioutput":
            # The Criterion initially generates one output for each y row and
            # y column, with a bunch of np.nan to indicate samples not in the
            # node. These values are then processed by predict to yield a
            # single output.
            n_raw_outputs = n_rows + n_cols
            if self.prediction_weights == "raw":
                self.n_outputs_ = n_raw_outputs
        else:
            n_raw_outputs = self.n_outputs_

        self._n_raw_outputs = n_raw_outputs

        if self.bipartite_adapter == "global_single_output":
            if self.criterion != "squared_error":
                raise NotImplementedError(  # TODO
                    "bipartite_adapter='global_single_output' currently only "
                    "supports criterion='squared_error'. Received "
                    f"{self.criterion=!r}. Notice that 'squared_error' "
                    "corresponds to the Gini impurity when targets are binary."
                )
            if self.prediction_weights is not None:
                raise NotImplementedError(  # TODO
                    "prediction_weights are only implemented for "
                    "bipartite_adapter='local_multioutput' and must be set to "
                    "'None' otherwise."
                )

        # if bipartite_adapter='local_multioutput'
        elif _is_arraylike_not_scalar(self.prediction_weights):
            if self.prediction_weights.ndim != 1:
                raise ValueError("If an array, 'prediction_weights' must be "
                                 "one-dimensional. Received "
                                 f"{self.prediction_weights.shape = }.")
            if self.prediction_weights.size != n_raw_outputs:
                raise ValueError("If an array, prediction_weights must be of "
                                 f"length = {n_rows + n_cols = }. Received "
                                 f"{len(self.prediction_weights) = }.")

        # callable, "uniform", or "precomputed"
        elif self.prediction_weights not in ("raw", "leaf_uniform"):
            for Xi in X:
                check_similarity_matrix(Xi, symmetry_exception=True)

        if is_classifier(self):
            self.tree_ = Tree(
                self.n_features_in_,
                self.n_classes_,
                n_raw_outputs,
            )
        else:
            self.tree_ = Tree(
                self.n_features_in_,
                # TODO: tree shouldn't need this in this case
                np.array([1] * n_raw_outputs, dtype=np.intp),
                n_raw_outputs,
            )

        splitter = self._make_splitter(
            n_samples=(n_rows, n_cols),
            # FIXME
            n_outputs=1 if self.bipartite_adapter == "global_single_output" else (n_cols, n_rows),
            # n_outputs=n_raw_outputs,
            sparse=issparse(X),
            min_samples_leaf=min_samples_leaf,
            min_weight_leaf=min_weight_leaf,
            ax_max_features=(max_row_features, max_col_features),
            ax_min_samples_leaf=(min_rows_leaf, min_cols_leaf),
            ax_min_weight_leaf=(min_row_weight_leaf, min_col_weight_leaf),
            random_state=random_state,
        )

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder2D(
                splitter,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_leaf=min_weight_leaf,
                max_depth=max_depth,
                min_impurity_decrease=self.min_impurity_decrease,
                # Bipartite parameters
                min_rows_split=min_rows_split,
                min_rows_leaf=min_rows_leaf,
                min_row_weight_leaf=min_row_weight_leaf,
                min_cols_split=min_cols_split,
                min_cols_leaf=min_cols_leaf,
                min_col_weight_leaf=min_col_weight_leaf,
            )
        else:
            raise NotImplementedError("Pleaase set max_leaf_nodes=None")
            builder = BestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )

        if row_weight is None and col_weight is None:
            sample_weight = None
        elif row_weight is None:
            row_weight = np.ones(n_rows)
        elif col_weight is None:
            col_weight = np.ones(n_cols)

        sample_weight = sample_weight and np.hstack([row_weight, col_weight])

        builder.build(self.tree_, X, y, sample_weight)

        if self.n_outputs_ == 1 and is_classifier(self):
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        self._prune_tree()

        return self

    def _make_splitter(
        self,
        *,
        n_samples,
        n_outputs,
        sparse=False,
        min_samples_leaf=1,
        min_weight_leaf=0.,
        ax_max_features=None,
        ax_min_samples_leaf=1,
        ax_min_weight_leaf=0.,
        random_state=None,
    ):
        if isinstance(self.splitter, Splitter2D):
            # Make a deepcopy in case the splitter has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            return copy.deepcopy(self.splitter)

        bipartite_adapter = self.bipartite_adapter
        if isinstance(bipartite_adapter, str):
            bipartite_adapter = BIPARTITE_CRITERIA[bipartite_adapter]

        criterion = self.criterion
        if isinstance(criterion, str):
            if is_classifier(self):
                if self.bipartite_adapter == "local_multioutput":
                    raise NotImplementedError
                    # criterion = AXIS_CRITERIA_CLF[criterion]
                else:
                    criterion = CRITERIA_CLF[criterion]
            else:
                if self.bipartite_adapter == "local_multioutput":
                    criterion = AXIS_CRITERIA_REG[criterion]
                else:
                    criterion = CRITERIA_REG[criterion]
        else:
            criterion = copy.deepcopy(criterion)

        SPLITTERS = SPARSE_SPLITTERS if sparse else DENSE_SPLITTERS

        splitter = self.splitter
        # User is able to specify a splitter for each axis
        if not isinstance(splitter, (tuple, list)):
            splitter = [splitter, splitter]
        for ax in range(2):
            if isinstance(splitter[ax], str):
                splitter[ax] = SPLITTERS[splitter[ax]]
            else:  # is a Splitter instance
                splitter[ax] = copy.deepcopy(splitter[ax])

        splitter = make_2d_splitter(
            splitters=splitter,
            criteria=criterion,
            n_samples=n_samples,
            n_outputs=n_outputs,
            max_features=ax_max_features,
            min_samples_leaf=min_samples_leaf,
            min_weight_leaf=min_weight_leaf,
            ax_min_samples_leaf=ax_min_samples_leaf,
            ax_min_weight_leaf=ax_min_weight_leaf,
            random_state=random_state,
            criterion_wrapper_class=bipartite_adapter,
        )

        return splitter

    def predict(self, X, check_input=True):
        # FIXME: classification is still not working for bipartite_adapter
        #        local_multioutput
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)
        n_samples = X.shape[0]

        # TODO: weighted outputs for global_single_output adapter
        if self.bipartite_adapter == "local_multioutput":
            proba = self._weight_raw_predictions(X, proba)

        # Classification
        if is_classifier(self):
            if self.n_outputs_ == 1:
                return self.classes_.take(np.argmax(proba, axis=-1), axis=0)

            else:
                class_type = self.classes_[0].dtype
                predictions = np.zeros(
                    (n_samples, self.n_outputs_), dtype=class_type)
                for k in range(self.n_outputs_):
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(proba[:, k], axis=1), axis=0
                    )

                return predictions

        # Regression
        else:
            if self.n_outputs_ == 1:
                return proba[:, 0]
            else:
                return proba[:, :, 0]

    def _validate_X_predict(self, X, check_input):
        """Validate the training data on predict (probabilities)."""
        if _X_is_multipartite(X):
            X = row_cartesian_product(X)

        return super()._validate_X_predict(X, check_input)

    def _weight_raw_predictions(self, X, pred):
        # Check prediction weights
        if self.prediction_weights == "raw":
            return pred
        if self.prediction_weights == "leaf_uniform":
            return np.nanmean(pred[:, :self._n_rows_fit], axis=1)
        if self.prediction_weights in (None, "uniform"):
            # Lines with known instances will only use the output corresponding
            # to its index in the training set.
            weights = (X == 1.).astype(DTYPE)
        elif self.prediction_weights == "precomputed":
            # TODO: check_precomputed
            weights = X
        elif _is_arraylike_not_scalar(self.prediction_weights):
            weights = self.prediction_weights.reshape(1, -1)

        else:  # callable
            weights = self.prediction_weights(X)
            if weights.shape != X.shape:
                raise ValueError(
                    "Callable prediction_weights must return an array with the"
                    " same shape as its input. Received "
                    f"{weights.shape=} from {X.shape=}."
                )

        weights = weights.reshape(pred.shape)
        row_weights, col_weights = np.hsplit(weights, [self._n_rows_fit])
        row_pred, col_pred = np.hsplit(pred, [self._n_rows_fit])
        row_weights = _normalize_weights(row_weights, row_pred)
        col_weights = _normalize_weights(col_weights, col_pred)

        row_final_pred = np.nansum(row_weights * row_pred, axis=1)
        col_final_pred = np.nansum(col_weights * col_pred, axis=1)

        return (row_final_pred + col_final_pred) / 2


# =============================================================================
# Public estimators
# =============================================================================


class BipartiteDecisionTreeRegressor(
    MultipartiteRegressorMixin,
    BaseBipartiteDecisionTree,
    DecisionTreeRegressor,
):
    """Decision tree regressor tailored to bipartite input.

    Implements optimized global single output (GSO) and multi-output (GMO)
    trees for interaction prediction. The latter is proposed by [1] under the
    name of Predictive Bi-Clustering  Trees. The former implements an optimzied
    algorithm for growing GSO trees, which consider concatenated pairs of row
    and column instances in a bipartite dataset as the actual intances.

    GSO trees (bipartite_adapter="global_single_output") will yield the exactly
    same tree structure as if all possible combinations of row and column
    instances were provided to a usual sklearn.DecisionTreeRegressor, but
    in much sorter time, by a whole factor of the number of instances.

    TODO: improve.

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

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    prediction_weights : {"uniform", "x", "raw"}, float, 1D-array or callable,
                         default="uniform"
        Determines how to compute the final predicted value. Initially, all
        predictions for each row and column instance from the training set that
        share the leaf node with the predicting sample are obtained.

        - "raw" instructs to return this vector, with a value for each training
          row and training column, and `np.nan` for instances not in the same
          leaf.
        - "leaf_uniform" returns the mean value of the leaf. Corresponds to
          the :math:`GMO_{SA}` approach described in [1].
        - "uniform" returns the mean value of the leaf for new instances but,
          for instances present in the training set, it uses the leaf's row or
          column mean corresponding to it. Known instances are recognized by a
          a similarity value of 1. This is the main approach presented by [1],
          named simply global multi-output (GMO).

        Other options return the weighted average of the leaf values:

        - A 1D-array may be provided to specify training sample weights
          explicitly, with weights for training row samples followed by weights
          for training column samples (length==`sum(y_train.shape)`).
        - "precomputed" instructs the estimator to consider x values as
          similarities to each row and column sample in the training set (row
          similarities followed by column similarities), and thus use it as
          weights to average the leaf outputs.
        - A callable, if provided, takes all the X being predicted and must
          return an array of weights for each predicting sample with the same
          shape as the X array given to predict().

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

    max_row_features_ : int
        The inferred value of max_row_features.

    max_col_features_ : int
        The inferred value of max_col_features.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    n_row_features_in_ : int
        Number of row features seen during :term:`fit`.

    n_col_features_in_ : int
        Number of column features seen during :term:`fit`.

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
    .. [1] :doi:`Global Multi-Output Decision Trees for interaction prediction \
       <doi.org/10.1007/s10994-018-5700-x>`
       Pliakos, Geurts and Vens, 2018

    .. [2] :doi:`Drug-target interaction prediction with tree-ensemble \
           learning and output space reconstruction \
           <doi.org/10.1186/s12859-020-3379-z>`
           Pliakos and Vens, 2020

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)


        cross_val_score(regressor, X, y, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([-0.39..., -0.46...,  0.02...,  0.06..., -0.50...,
           0.16...,  0.11..., -0.73..., -0.30..., -0.00...])
    """
    _parameter_constraints: dict = {
        **BaseBipartiteDecisionTree._parameter_constraints,
        "criterion": [
            StrOptions({"squared_error"}),
            Hidden(StrOptions({"friedman_mse", "absolute_error", "poisson"})),
            Hidden(Criterion),
        ],
    }

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
        min_rows_split=1,
        min_cols_split=1,
        min_rows_leaf=1,
        min_cols_leaf=1,
        min_row_weight_fraction_leaf=0.0,
        min_col_weight_fraction_leaf=0.0,
        max_row_features=None,
        max_col_features=None,
        bipartite_adapter="global_single_output",
        prediction_weights=None,
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
        )

    def fit(self, X, y, row_weight=None, col_weight=None, check_input=True):
        """Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : list-like of {array-like, sparse matrix} of shapes (n_axis_samples,
            n_axis_features).
            The training input samples for each axis. Internally, it will be
            converted to ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_row_samples, n_col_samples)
            The target values (real . Use ``dtype=np.float64`` and
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
            row_weight=row_weight,
            col_weight=col_weight,
            check_input=check_input,
        )
        return self


class BipartiteExtraTreeRegressor(BipartiteDecisionTreeRegressor):
    """Extremetly randomizes tree regressor tailored to bipartite input.

    Implements optimized global single output (GSO) and multi-output (GMO)
    trees for interaction prediction. The latter is proposed by [1] under the
    name of Predictive Bi-Clustering  Trees. The former implements an optimzied
    algorithm for growing GSO trees, which consider concatenated pairs of row
    and column instances in a bipartite dataset as the actual intances.

    GSO trees (bipartite_adapter="global_single_output") will yield the exactly
    same tree structure as if all possible combinations of row and column
    instances were provided to a usual sklearn.DecisionTreeRegressor, but
    in much sorter time, by a whole factor of the number of instances.

    TODO: improve.

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

    splitter : {"best", "random"}, default="random"
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

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    prediction_weights : {"uniform", "x", "raw"}, float, 1D-array or callable,
                         default="uniform"
        Determines how to compute the final predicted value. Initially, all
        predictions for each row and column instance from the training set that
        share the leaf node with the predicting sample are obtained.

        - "raw" instructs to return this vector, with a value for each training
          row and training column, and `np.nan` for instances not in the same
          leaf.
        - "leaf_uniform" returns the mean value of the leaf. Corresponds to
          the :math:`GMO_{SA}` approach described in [1].
        - "uniform" returns the mean value of the leaf for new instances but,
          for instances present in the training set, it uses the leaf's row or
          column mean corresponding to it. Known instances are recognized by a
          a similarity value of 1. This is the main approach presented by [1],
          named simply global multi-output (GMO).

        Other options return the weighted average of the leaf values:

        - A 1D-array may be provided to specify training sample weights
          explicitly, with weights for training row samples followed by weights
          for training column samples (length==`sum(y_train.shape)`).
        - "precomputed" instructs the estimator to consider x values as
          similarities to each row and column sample in the training set (row
          similarities followed by column similarities), and thus use it as
          weights to average the leaf outputs.
        - A callable, if provided, takes all the X being predicted and must
          return an array of weights for each predicting sample with the same
          shape as the X array given to predict().

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

    max_row_features_ : int
        The inferred value of max_row_features.

    max_col_features_ : int
        The inferred value of max_col_features.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    n_row_features_in_ : int
        Number of row features seen during :term:`fit`.

    n_col_features_in_ : int
        Number of column features seen during :term:`fit`.

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
    .. [1] :doi:`Global Multi-Output Decision Trees for interaction prediction \
       <doi.org/10.1007/s10994-018-5700-x>`
       Pliakos, Geurts and Vens, 2018

    .. [2] :doi:`Drug-target interaction prediction with tree-ensemble \
           learning and output space reconstruction \
           <doi.org/10.1186/s12859-020-3379-z>`
           Pliakos and Vens, 2020

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)


        cross_val_score(regressor, X, y, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([-0.39..., -0.46...,  0.02...,  0.06..., -0.50...,
           0.16...,  0.11..., -0.73..., -0.30..., -0.00...])
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
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        # Bipartite parameters:
        min_rows_split=1,
        min_cols_split=1,
        min_rows_leaf=1,
        min_cols_leaf=1,
        min_row_weight_fraction_leaf=0.0,
        min_col_weight_fraction_leaf=0.0,
        max_row_features=None,
        max_col_features=None,
        bipartite_adapter="global_single_output",
        prediction_weights=None,
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
        )
