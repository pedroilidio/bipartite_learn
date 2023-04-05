"""Gradient Boosted Regression Trees.

This module contains methods for fitting gradient boosted regression trees for
both classification and regression.

The module structure is the following:

- The ``BaseGradientBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ in the concrete ``LossFunction`` used.

- ``GradientBoostingClassifier`` implements gradient boosting for
  classification problems.

- ``GradientBoostingRegressor`` implements gradient boosting for
  regression problems.
"""
# Author: Pedro Ilídio
# Adapted from scikit-learn.
# License: BSD 3 clause

from abc import ABCMeta
from abc import abstractmethod
from numbers import Integral, Real
import warnings

from sklearn.ensemble._base import BaseEnsemble
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.base import is_classifier
from sklearn.utils import deprecated

from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.ensemble._gradient_boosting import predict_stage
from sklearn.ensemble._gradient_boosting import _random_sample_mask

import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from time import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.ensemble import _gb_losses

from sklearn.utils import check_array, check_random_state, column_or_1d
from sklearn.utils._tags import _safe_tags
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.exceptions import NotFittedError
from sklearn.ensemble._gb import VerboseReporter
from sklearn.ensemble._gb import BaseGradientBoosting

from ..tree import BipartiteDecisionTreeRegressor
from ..base import BaseBipartiteEstimator
from ..melter import melt_multipartite_dataset
from ..wrappers import GlobalSingleOutputWrapper
from ..utils import _check_multipartite_sample_weight
from ..model_selection import train_test_split_nd


__all__ = [
    "BipartiteGradientBoostingClassifier",
    "BipartiteGradientBoostingRegressor",
]


def _expand_sample_weight(sample_weight, n_rows):
    if sample_weight is None:
        return None
    row_weight = sample_weight[:n_rows]
    col_weight = sample_weight[n_rows:]
    return (row_weight[:, np.newaxis] * col_weight) \
        .reshape(-1).astype(sample_weight.dtype)


class BaseBipartiteGradientBoosting(
    BaseBipartiteEstimator,
    BaseGradientBoosting,
    BaseEnsemble,
    metaclass=ABCMeta,
):
    """Abstract base class for Bipartite Gradient Boosting."""

    _parameter_constraints: dict = {
        **BaseGradientBoosting._parameter_constraints,
        **BipartiteDecisionTreeRegressor._parameter_constraints,
    }
    _parameter_constraints.pop("splitter")

    @abstractmethod
    def __init__(
        self,
        *,
        loss,
        learning_rate,
        n_estimators,
        init,
        subsample,
        random_state,
        alpha=0.9,
        verbose=0,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        # Estimator params
        criterion,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_leaf_nodes,
        min_impurity_decrease,
        ccp_alpha,
        max_features=None,
        # Estimator's bipartite params
        min_rows_split=1,
        min_cols_split=1,
        min_rows_leaf=1,
        min_cols_leaf=1,
        min_row_weight_fraction_leaf=0.0,
        min_col_weight_fraction_leaf=0.0,
        max_row_features=None,
        max_col_features=None,
        bipartite_adapter="gmosa",
        prediction_weights=None,
    ):
        # FIXME: skipping BaseGradientBoosting.__init__ to keep using estimator
        # params and _make_estimator from grandparent BaseEnsemble, which were
        # overridden by BaseGradientBoosting.
        # super(BaseGradientBoosting, self).__init__(
        BaseEnsemble.__init__(
            self,
            estimator=BipartiteDecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                # Bipartite parameters:
                "min_rows_split",
                "min_cols_split",
                "min_rows_leaf",
                "min_cols_leaf",
                "min_row_weight_fraction_leaf",
                "min_col_weight_fraction_leaf",
                "max_row_features",
                "max_col_features",
                "bipartite_adapter",
                "prediction_weights",
            ),
        )

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.subsample = subsample
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

        # Estimator params
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

        # Bipartite estimator params
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

    def _make_estimator(self, append=True, random_state=None):
        # FIXME: skipping BaseGradientBoosting._make_estimator to keep using
        # _make_estimator from grandparent BaseEnsemble, which was
        # overridden by BaseGradientBoosting.
        # super(BaseGradientBoosting, self)._make_estimator()
        return BaseEnsemble._make_estimator(
            self, append, random_state=random_state,
        )

    def _fit_stage(
        self,
        i,
        X,
        y,
        raw_predictions,
        sample_weight,
        sample_mask,
        random_state,
        X_csc=None,
        X_csr=None,
    ):
        """Fit another stage of ``_n_classes`` trees to the boosting model."""

        assert sample_mask.dtype == bool
        loss = self._loss
        X_molten, y_molten = melt_multipartite_dataset(X, y)
        n_rows = X[0].shape[0]
        # FIXME: avoid calling _expand_sample_weight here, once for each tree
        expanded_sample_weight = _expand_sample_weight(sample_weight, n_rows)

        # Need to pass a copy of raw_predictions to negative_gradient()
        # because raw_predictions is partially updated at the end of the loop
        # in update_terminal_regions(), and gradients need to be evaluated at
        # iteration i - 1.
        raw_predictions_copy = raw_predictions.copy()

        for k in range(loss.K):
            if loss.is_multi_class:
                y_molten_k = np.array(y_molten == k, dtype=np.float64)
            else:
                y_molten_k = y_molten

            residual = loss.negative_gradient(
                y_molten_k,
                raw_predictions_copy,
                k=k,
                sample_weight=expanded_sample_weight,
            )

            # Induce tree on residuals
            tree = self._make_estimator(
                random_state=random_state,
                append=False,
            )

            if self.subsample < 1.0:
                # Make sure to not modify sample_weight
                masked_sample_weight = (
                    sample_weight * sample_mask.astype(np.float64)
                )
            else:
                masked_sample_weight = sample_weight

            X = X_csr if X_csr[0] is not None else X
            tree.fit(
                X,
                residual.reshape(y.shape),
                sample_weight=masked_sample_weight,
                check_input=False,
            )

            # update tree leaves
            loss.update_terminal_regions(
                tree.tree_,
                X_molten,
                y_molten_k,
                residual,
                raw_predictions,
                _expand_sample_weight(masked_sample_weight, n_rows),
                _expand_sample_weight(sample_mask, n_rows),
                learning_rate=self.learning_rate,
                k=k,
            )

            # add tree to ensemble
            self.estimators_[i, k] = tree

        return raw_predictions

    # TODO: simplify
    def _check_params(self):
        # TODO(1.3): Remove
        if self.loss == "deviance":
            warnings.warn(
                "The loss parameter name 'deviance' was deprecated in v1.1 and will be "
                "removed in version 1.3. Use the new parameter name 'log_loss' which "
                "is equivalent.",
                FutureWarning,
            )
            loss_class = (
                _gb_losses.MultinomialDeviance
                if len(self.classes_) > 2
                else _gb_losses.BinomialDeviance
            )
        elif self.loss == "log_loss":
            loss_class = (
                _gb_losses.MultinomialDeviance
                if len(self.classes_) > 2
                else _gb_losses.BinomialDeviance
            )
        else:
            loss_class = _gb_losses.LOSS_FUNCTIONS[self.loss]

        if is_classifier(self):
            self._loss = loss_class(self.n_classes_)
        elif self.loss in ("huber", "quantile"):
            self._loss = loss_class(self.alpha)
        else:
            self._loss = loss_class()

        # TODO: extract from self.estimator_
        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classifier(self):
                    max_features = max(1, int(np.sqrt(self.n_features_in_)))
                else:
                    max_features = self.n_features_in_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            else:  # self.max_features == "log2"
                max_features = max(1, int(np.log2(self.n_features_in_)))
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, Integral):
            max_features = self.max_features
        else:  # float
            max_features = max(1, int(self.max_features * self.n_features_in_))

        self.max_features_ = max_features

    def fit(self, X, y, sample_weight=None, monitor=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        y : array-like of shape (n_samples,)
            Target values (strings or integers in classification, real numbers
            in regression)
            For classification, labels must correspond to classes.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        monitor : callable, default=None
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspect, and
            snapshoting.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()
        self._validate_estimator()

        if not self.warm_start:
            self._clear_state()

        # Check input
        # Since check_array converts both X and y to the same dtype, but the
        # trees use different types for X and y, checking them separately.

        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc", "coo"],
            dtype=DTYPE,
            # multi_output=True,  # TODO: multipartite multioutput (3D y)
        )

        sample_weight_is_none = sample_weight is None

        sample_weight = _check_multipartite_sample_weight(sample_weight, X)

        # y = column_or_1d(y, warn=True)

        if is_classifier(self):
            y = self._validate_y(
                # TODO: modify _validate_y to avoid expanding here.
                y, _expand_sample_weight(sample_weight, y.shape[0]),
            ).reshape(y.shape)
        else:
            y = self._validate_y(y)

        self._check_params()

        if self.n_iter_no_change is not None:
            stratify = y if is_classifier(self) else None
            (
                X,
                X_val,
                y,
                y_val,
                sample_weight,
                sample_weight_val,
            ) = train_test_split_nd(
                X,
                y,
                sample_weight,
                random_state=self.random_state,
                test_size=self.validation_fraction,
                stratify=stratify,
            )
            if is_classifier(self):
                if self._n_classes != np.unique(y).shape[0]:
                    # We choose to error here. The problem is that the init
                    # estimator would be trained on y, which has some missing
                    # classes now, so its predictions would not have the
                    # correct shape.
                    raise ValueError(
                        "The training data after the early stopping split "
                        "is missing some classes. Try using another random "
                        "seed."
                    )
        else:
            X_val = y_val = sample_weight_val = None

        if not self._is_initialized():
            # init state
            self._init_state()
            n_samples = y.size

            # fit initial model and initialize raw predictions
            if self.init_ == "zero":
                raw_predictions = np.zeros(
                    shape=(n_samples, self._loss.K), dtype=np.float64
                )
            else:
                # XXX clean this once we have a support_sample_weight tag
                if sample_weight_is_none:
                    self.init_.fit(X, y)
                else:
                    msg = (
                        "The initial estimator {} does not support sample "
                        "weights.".format(self.init_.__class__.__name__)
                    )
                    try:
                        self.init_.fit(X, y, sample_weight=sample_weight)
                    except TypeError as e:
                        if "unexpected keyword argument 'sample_weight'" in str(e):
                            # regular estimator without SW support
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise
                    except ValueError as e:
                        if (
                            "pass parameters to specific steps of "
                            "your pipeline using the "
                            "stepname__parameter"
                            in str(e)
                        ):  # pipeline
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise

                raw_predictions = self._loss.get_init_raw_predictions(
                    X, self.init_,
                )

            begin_at_stage = 0

            # The rng state must be preserved if warm_start is True
            self._rng = check_random_state(self.random_state)

        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError(
                    "n_estimators=%d must be larger or equal to "
                    "estimators_.shape[0]=%d when "
                    "warm_start==True" % (
                        self.n_estimators, self.estimators_.shape[0])
                )
            begin_at_stage = self.estimators_.shape[0]
            # The requirements of _raw_predict
            # are more constrained than fit. It accepts only CSR
            # matrices. Finite values have already been checked in _validate_data.
            for ax in range(len(X)):
                X[ax] = check_array(
                    X[ax],
                    dtype=DTYPE,
                    order="C",
                    accept_sparse="csr",
                    force_all_finite=False,
                )
            raw_predictions = self._raw_predict(X)
            self._resize_state()

        # fit the boosting stages
        n_stages = self._fit_stages(
            X,
            y,
            raw_predictions,
            sample_weight,
            self._rng,
            X_val,
            y_val,
            sample_weight_val,
            begin_at_stage,
            monitor,
        )

        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, "oob_improvement_"):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        self.n_estimators_ = n_stages
        return self

    def _fit_stages(
        self,
        X,
        y,
        raw_predictions,
        sample_weight,
        random_state,
        X_val,
        y_val,
        sample_weight_val,
        begin_at_stage=0,
        monitor=None,
    ):
        """Iteratively fits the stages.

        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        n_rows, n_cols = y.shape
        y_molten = y.reshape(-1)
        if y_val is not None:
            y_val_molten = y_val.reshape(-1)
        expanded_sample_weight = _expand_sample_weight(sample_weight, n_rows)
        expanded_sample_weight_val = _expand_sample_weight(
            sample_weight_val, n_rows,
        )

        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_rows + n_cols,), dtype=bool)
        eff_subsample = np.sqrt(self.subsample)
        n_rows_inbag = max(1, int(eff_subsample * n_rows))
        n_cols_inbag = max(1, int(eff_subsample * n_cols))
        loss_ = self._loss

        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc, X_csr = [], []
        for Xi in X:
            X_csc.append(csc_matrix(Xi) if issparse(Xi) else None)
            X_csr.append(csr_matrix(Xi) if issparse(Xi) else None)

        if self.n_iter_no_change is not None:
            loss_history = np.full(self.n_iter_no_change, np.inf)
            # We create a generator to get the predictions for X_val after
            # the addition of each successive stage
            y_val_pred_iter = self._staged_raw_predict(
                X_val, check_input=False,
            )

        # perform boosting iterations
        i = begin_at_stage
        for i in range(begin_at_stage, self.n_estimators):

            # subsampling
            if do_oob:
                row_sample_mask = _random_sample_mask(
                    n_rows, n_rows_inbag, random_state,
                )
                col_sample_mask = _random_sample_mask(
                    n_cols, n_cols_inbag, random_state,
                )
                sample_mask = np.hstack((row_sample_mask, col_sample_mask))

                expanded_sample_mask = _expand_sample_weight(
                    sample_mask, n_rows,
                )
                # OOB score before adding this stage
                old_oob_score = loss_(
                    y_molten[~expanded_sample_mask],
                    raw_predictions[~expanded_sample_mask],
                    expanded_sample_weight[~expanded_sample_mask],
                )

            # fit next stage of trees
            raw_predictions = self._fit_stage(
                i,
                X,
                y,
                raw_predictions,
                sample_weight,
                sample_mask,
                random_state,
                X_csc,
                X_csr,
            )

            # track deviance (= loss)
            if do_oob:
                self.train_score_[i] = loss_(
                    y_molten[expanded_sample_mask],
                    raw_predictions[expanded_sample_mask],
                    expanded_sample_weight[expanded_sample_mask],
                )
                self.oob_improvement_[i] = old_oob_score - loss_(
                    y_molten[~expanded_sample_mask],
                    raw_predictions[~expanded_sample_mask],
                    expanded_sample_weight[~expanded_sample_mask],
                )
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = loss_(
                    y_molten, raw_predictions, expanded_sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(i, self)

            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break

            # We also provide an early stopping based on the score from
            # validation set (X_val, y_val), if n_iter_no_change is set
            if self.n_iter_no_change is not None:
                # By calling next(y_val_pred_iter), we get the predictions
                # for X_val after the addition of the current stage
                validation_loss = loss_(
                    y_val_molten,
                    next(y_val_pred_iter),
                    expanded_sample_weight_val,
                )

                # Require validation_score to be better (less) than at least
                # one of the last n_iter_no_change evaluations
                if np.any(validation_loss + self.tol < loss_history):
                    loss_history[i % len(loss_history)] = validation_loss
                else:
                    break

        return i + 1

    def _init_state(self):
        """Initialize model state and allocate model state data structures."""
        super()._init_state()

        if self.init_ in ("zeros", None):
            return
        if not _safe_tags(self.init_).get("n_partite"):
            # if self.init_ is a monopartite estimator
            self.init_ = GlobalSingleOutputWrapper(self.init_)
    
    def _validate_X_predict(self, X, check_input=True):
        return self.estimators_[0, 0]._validate_X_predict(X, check_input)

    # def _raw_predict_init(self, X):
    #     """Check input and compute raw predictions of the init estimator."""
    #     self._check_initialized()
    #     X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
    #     if self.init_ == "zero":
    #         raw_predictions = np.zeros(
    #             shape=(X.shape[0], self._loss.K), dtype=np.float64
    #         )
    #     else:
    #         raw_predictions = self._loss.get_init_raw_predictions(X, self.init_).astype(
    #             np.float64
    #         )
    #     return raw_predictions

    # def _raw_predict(self, X):
    #     """Return the sum of the trees raw predictions (+ init estimator)."""
    #     raw_predictions = self._raw_predict_init(X)
    #     predict_stages(self.estimators_, X, self.learning_rate, raw_predictions)
    #     return raw_predictions

    # def _staged_raw_predict(self, X, check_input=True):
    #     """Compute raw predictions of ``X`` for each iteration.

    #     This method allows monitoring (i.e. determine error on testing set)
    #     after each stage.

    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix} of shape (n_samples, n_features)
    #         The input samples. Internally, it will be converted to
    #         ``dtype=np.float32`` and if a sparse matrix is provided
    #         to a sparse ``csr_matrix``.

    #     check_input : bool, default=True
    #         If False, the input arrays X will not be checked.

    #     Returns
    #     -------
    #     raw_predictions : generator of ndarray of shape (n_samples, k)
    #         The raw predictions of the input samples. The order of the
    #         classes corresponds to that in the attribute :term:`classes_`.
    #         Regression and binary classification are special cases with
    #         ``k == 1``, otherwise ``k==n_classes``.
    #     """
    #     if check_input:
    #         X = self._validate_data(
    #             X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
    #         )
    #     raw_predictions = self._raw_predict_init(X)
    #     for i in range(self.estimators_.shape[0]):
    #         predict_stage(self.estimators_, i, X, self.learning_rate, raw_predictions)
    #         yield raw_predictions.copy()

    # def apply(self, X):
    #     """Apply trees in the ensemble to X, return leaf indices.

    #     .. versionadded:: 0.17

    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix} of shape (n_samples, n_features)
    #         The input samples. Internally, its dtype will be converted to
    #         ``dtype=np.float32``. If a sparse matrix is provided, it will
    #         be converted to a sparse ``csr_matrix``.

    #     Returns
    #     -------
    #     X_leaves : array-like of shape (n_samples, n_estimators, n_classes)
    #         For each datapoint x in X and for each tree in the ensemble,
    #         return the index of the leaf x ends up in each estimator.
    #         In the case of binary classification n_classes is 1.
    #     """

    #     self._check_initialized()
    #     X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

    #     # n_classes will be equal to 1 in the binary classification or the
    #     # regression case.
    #     n_estimators, n_classes = self.estimators_.shape
    #     leaves = np.zeros((X.shape[0], n_estimators, n_classes))

    #     for i in range(n_estimators):
    #         for j in range(n_classes):
    #             estimator = self.estimators_[i, j]
    #             leaves[:, i, j] = estimator.apply(X, check_input=False)

    #     return leaves


# XXX
# TODO: remove extra methods (also from Regressor)
class BipartiteGradientBoostingClassifier(
    ClassifierMixin,
    BaseBipartiteGradientBoosting,
):
    """Gradient Boosting for classification.

    This algorithm builds an additive model in a forward stage-wise fashion; it
    allows for the optimization of arbitrary differentiable loss functions. In
    each stage ``n_classes_`` regression trees are fit on the negative gradient
    of the loss function, e.g. binary or multiclass log loss. Binary
    classification is a special case where only a single regression tree is
    induced.

    :class:`sklearn.ensemble.HistGradientBoostingClassifier` is a much faster
    variant of this algorithm for intermediate datasets (`n_samples >= 10_000`).

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'log_loss', 'deviance', 'exponential'}, default='log_loss'
        The loss function to be optimized. 'log_loss' refers to binomial and
        multinomial deviance, the same as used in logistic regression.
        It is a good choice for classification with probabilistic outputs.
        For loss 'exponential', gradient boosting recovers the AdaBoost algorithm.

        .. deprecated:: 1.1
            The loss 'deviance' was deprecated in v1.1 and will be removed in
            version 1.3. Use `loss='log_loss'` which is equivalent.

    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        Values must be in the range `[0.0, inf)`.

    n_estimators : int, default=100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        Values must be in the range `[1, inf)`.

    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
        Values must be in the range `(0.0, 1.0]`.

    criterion : {'friedman_mse', 'squared_error'}, default='friedman_mse'
        The function to measure the quality of a split. Supported criteria are
        'friedman_mse' for the mean squared error with improvement score by
        Friedman, 'squared_error' for mean squared error. The default value of
        'friedman_mse' is generally the best as it can provide a better
        approximation in some cases.

        .. versionadded:: 0.18

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, values must be in the range `[2, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and `min_samples_split`
          will be `ceil(min_samples_split * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0)` and `min_samples_leaf`
          will be `ceil(min_samples_leaf * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
        Values must be in the range `[0.0, 0.5]`.

    max_depth : int or None, default=3
        Maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        If int, values must be in the range `[1, inf)`.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        Values must be in the range `[0.0, inf)`.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    init : estimator or 'zero', default=None
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide :meth:`fit` and :meth:`predict_proba`. If
        'zero', the initial raw predictions are set to zero. By default, a
        ``DummyEstimator`` predicting the classes priors is used.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each Tree estimator at each
        boosting iteration.
        In addition, it controls the random permutation of the features at
        each split (see Notes for more details).
        It also controls the random splitting of the training data to obtain a
        validation set if `n_iter_no_change` is not None.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    max_features : {'auto', 'sqrt', 'log2'}, int or float, default=None
        The number of features to consider when looking for the best split:

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and the features
          considered at each split will be `max(1, int(max_features * n_features_in_))`.
        - If 'auto', then `max_features=sqrt(n_features)`.
        - If 'sqrt', then `max_features=sqrt(n_features)`.
        - If 'log2', then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.
        Values must be in the range `[0, inf)`.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        Values must be in the range `[2, inf)`.
        If `None`, then unlimited number of leaf nodes.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Values must be in the range `(0.0, 1.0)`.
        Only used if ``n_iter_no_change`` is set to an integer.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations. The split is stratified.
        Values must be in the range `[1, inf)`.

        .. versionadded:: 0.20

    tol : float, default=1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.
        Values must be in the range `[0.0, inf)`.

        .. versionadded:: 0.20

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.
        Values must be in the range `[0.0, inf)`.
        See :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

        .. versionadded:: 0.20

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    oob_improvement_ : ndarray of shape (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
        Only available if ``subsample < 1.0``

    train_score_ : ndarray of shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

        .. deprecated:: 1.1
             Attribute `loss_` was deprecated in version 1.1 and will be
            removed in 1.3.

    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor of \
            shape (n_estimators, ``loss_.K``)
        The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
        classification, otherwise n_classes.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_classes_ : int
        The number of classes.

    max_features_ : int
        The inferred value of max_features.

    See Also
    --------
    HistGradientBoostingClassifier : Histogram-based Gradient Boosting
        Classification Tree.
    sklearn.tree.DecisionTreeClassifier : A decision tree classifier.
    RandomForestClassifier : A meta-estimator that fits a number of decision
        tree classifiers on various sub-samples of the dataset and uses
        averaging to improve the predictive accuracy and control over-fitting.
    AdaBoostClassifier : A meta-estimator that begins by fitting a classifier
        on the original dataset and then fits additional copies of the
        classifier on the same dataset where the weights of incorrectly
        classified instances are adjusted such that subsequent classifiers
        focus more on difficult cases.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.

    Examples
    --------
    The following example shows how to fit a gradient boosting classifier with
    100 decision stumps as weak learners.

    >>> from sklearn.datasets import make_hastie_10_2
    >>> from sklearn.ensemble import GradientBoostingClassifier

    >>> X, y = make_hastie_10_2(random_state=0)
    >>> X_train, X_test = X[:2000], X[2000:]
    >>> y_train, y_test = y[:2000], y[2000:]

    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    ...     max_depth=1, random_state=0).fit(X_train, y_train)
    >>> clf.score(X_test, y_test)
    0.913...
    """

    # TODO(sklearn 1.3): remove "deviance"
    _parameter_constraints: dict = {
        **BaseBipartiteGradientBoosting._parameter_constraints,
        "loss": [
            StrOptions({"log_loss", "deviance", "exponential"},
                       deprecated={"deviance"})
        ],
        "init": [StrOptions({"zero"}), None, HasMethods(["fit", "predict_proba"])],
    }

    def __init__(
        self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_gso",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
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
        bipartite_adapter="gmosa",
        prediction_weights=None,
    ):

        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
            # Bipartite parameters
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

    def _validate_y(self, y, sample_weight):
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
        if n_trim_classes < 2:
            raise ValueError(
                "y contains %d class after sample_weight "
                "trimmed classes with zero weights, while a "
                "minimum of 2 classes are required." % n_trim_classes
            )
        self._n_classes = len(self.classes_)
        # expose n_classes_ attribute
        self.n_classes_ = self._n_classes
        return y

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        score : ndarray of shape (n_samples, n_classes) or (n_samples,)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            order of the classes corresponds to that in the attribute
            :term:`classes_`. Regression and binary classification produce an
            array of shape (n_samples,).
        """
        X = self._validate_data(
            X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
        )
        raw_predictions = self._raw_predict(X)
        if raw_predictions.shape[1] == 1:
            return raw_predictions.ravel()
        return raw_predictions

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Yields
        ------
        score : generator of ndarray of shape (n_samples, k)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        yield from self._staged_raw_predict(X)

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        X = self._validate_X_predict(X)
        raw_predictions = self.decision_function(X)
        encoded_labels = self._loss._raw_prediction_to_decision(
            raw_predictions)
        return self.classes_.take(encoded_labels, axis=0)

    def staged_predict(self, X):
        """Predict class at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        X = self._validate_X_predict(X)
        for raw_predictions in self._staged_raw_predict(X):
            encoded_labels = self._loss._raw_prediction_to_decision(
                raw_predictions)
            yield self.classes_.take(encoded_labels, axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.
        """
        X = self._validate_X_predict(X)
        raw_predictions = self.decision_function(X)
        try:
            return self._loss._raw_prediction_to_proba(raw_predictions)
        except NotFittedError:
            raise
        except AttributeError as e:
            raise AttributeError(
                "loss=%r does not support predict_proba" % self.loss
            ) from e

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.
        """
        proba = self.predict_proba(X)
        return np.log(proba)

    def staged_predict_proba(self, X):
        """Predict class probabilities at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        X = self._validate_X_predict(X)
        try:
            for raw_predictions in self._staged_raw_predict(X):
                yield self._loss._raw_prediction_to_proba(raw_predictions)
        except NotFittedError:
            raise
        # TODO: use @available_if?
        except AttributeError as e:
            raise AttributeError(
                "loss=%r does not support predict_proba" % self.loss
            ) from e


class BipartiteGradientBoostingRegressor(
    RegressorMixin,
    BaseBipartiteGradientBoosting,
):
    """Gradient Boosting for regression.

    This estimator builds an additive model in a forward stage-wise fashion; it
    allows for the optimization of arbitrary differentiable loss functions. In
    each stage a regression tree is fit on the negative gradient of the given
    loss function.

    :class:`sklearn.ensemble.HistGradientBoostingRegressor` is a much faster
    variant of this algorithm for intermediate datasets (`n_samples >= 10_000`).

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'squared_error', 'absolute_error', 'huber', 'quantile'}, \
            default='squared_error'
        Loss function to be optimized. 'squared_error' refers to the squared
        error for regression. 'absolute_error' refers to the absolute error of
        regression and is a robust loss function. 'huber' is a
        combination of the two. 'quantile' allows quantile regression (use
        `alpha` to specify the quantile).

    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        Values must be in the range `[0.0, inf)`.

    n_estimators : int, default=100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        Values must be in the range `[1, inf)`.

    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
        Values must be in the range `(0.0, 1.0]`.

    criterion : {'friedman_mse', 'squared_error'}, default='friedman_mse'
        The function to measure the quality of a split. Supported criteria are
        "friedman_mse" for the mean squared error with improvement score by
        Friedman, "squared_error" for mean squared error. The default value of
        "friedman_mse" is generally the best as it can provide a better
        approximation in some cases.

        .. versionadded:: 0.18

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, values must be in the range `[2, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and `min_samples_split`
          will be `ceil(min_samples_split * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0)` and `min_samples_leaf`
          will be `ceil(min_samples_leaf * n_samples)`.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
        Values must be in the range `[0.0, 0.5]`.

    max_depth : int or None, default=3
        Maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        If int, values must be in the range `[1, inf)`.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        Values must be in the range `[0.0, inf)`.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    init : estimator or 'zero', default=None
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide :term:`fit` and :term:`predict`. If 'zero', the
        initial raw predictions are set to zero. By default a
        ``DummyEstimator`` is used, predicting either the average target value
        (for loss='squared_error'), or a quantile for the other losses.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each Tree estimator at each
        boosting iteration.
        In addition, it controls the random permutation of the features at
        each split (see Notes for more details).
        It also controls the random splitting of the training data to obtain a
        validation set if `n_iter_no_change` is not None.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    max_features : {'auto', 'sqrt', 'log2'}, int or float, default=None
        The number of features to consider when looking for the best split:

        - If int, values must be in the range `[1, inf)`.
        - If float, values must be in the range `(0.0, 1.0]` and the features
          considered at each split will be `max(1, int(max_features * n_features_in_))`.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    alpha : float, default=0.9
        The alpha-quantile of the huber loss function and the quantile
        loss function. Only if ``loss='huber'`` or ``loss='quantile'``.
        Values must be in the range `(0.0, 1.0)`.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.
        Values must be in the range `[0, inf)`.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        Values must be in the range `[2, inf)`.
        If None, then unlimited number of leaf nodes.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Values must be in the range `(0.0, 1.0)`.
        Only used if ``n_iter_no_change`` is set to an integer.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.
        Values must be in the range `[1, inf)`.

        .. versionadded:: 0.20

    tol : float, default=1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.
        Values must be in the range `[0.0, inf)`.

        .. versionadded:: 0.20

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.
        Values must be in the range `[0.0, inf)`.
        See :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    oob_improvement_ : ndarray of shape (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
        Only available if ``subsample < 1.0``

    train_score_ : ndarray of shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

        .. deprecated:: 1.1
             Attribute `loss_` was deprecated in version 1.1 and will be
            removed in 1.3.

    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor of shape (n_estimators, 1)
        The collection of fitted sub-estimators.

    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    max_features_ : int
        The inferred value of max_features.

    See Also
    --------
    HistGradientBoostingRegressor : Histogram-based Gradient Boosting
        Classification Tree.
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.
    sklearn.ensemble.RandomForestRegressor : A random forest regressor.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_regression(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> reg = GradientBoostingRegressor(random_state=0)
    >>> reg.fit(X_train, y_train)
    GradientBoostingRegressor(random_state=0)
    >>> reg.predict(X_test[1:2])
    array([-61...])
    >>> reg.score(X_test, y_test)
    0.4...
    """

    _parameter_constraints: dict = {
        **BaseBipartiteGradientBoosting._parameter_constraints,
        "loss": [StrOptions({"squared_error", "absolute_error", "huber", "quantile"})],
        "init": [StrOptions({"zero"}), None, HasMethods(["fit", "predict"])],
        "alpha": [Interval(Real, 0.0, 1.0, closed="neither")],
    }

    def __init__(
        self,
        *,
        loss="squared_error",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_gso",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=None,
        max_features=None,
        alpha=0.9,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
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
        bipartite_adapter="gmosa",
        prediction_weights=None,
    ):

        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            alpha=alpha,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
            # Bipartite parameters
            min_rows_split=min_rows_split,
            min_cols_split=min_cols_split,
            min_rows_leaf=min_rows_leaf,
            min_cols_leaf=min_cols_leaf,
            min_row_weight_fraction_leaf=min_row_weight_fraction_leaf,
            min_col_weight_fraction_leaf=min_col_weight_fraction_leaf,
            max_row_features=max_row_features,
            max_col_features=max_col_features,
            bipartite_adapter=bipartite_adapter,
        )

    def _validate_y(self, y, sample_weight=None):
        if y.dtype.kind == "O":
            y = y.astype(DOUBLE)
        return y

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        X = self._validate_X_predict(X)
        X = self._validate_data(
            X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
        )
        # In regression we can directly return the raw value from the trees.
        return self._raw_predict(X).ravel()

    def staged_predict(self, X):
        """Predict regression target at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted value of the input samples.
        """
        X = self._validate_X_predict(X)
        for raw_predictions in self._staged_raw_predict(X):
            yield raw_predictions.ravel()

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array-like of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
        """
        leaves = super().apply(X)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves
