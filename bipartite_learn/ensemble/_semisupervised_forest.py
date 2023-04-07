# TODO: Documentation.
# TODO: Classifiers.
from abc import ABCMeta
from sklearn.ensemble._forest import (
    BaseForest,
    ForestRegressor,
    ForestClassifier,
)
from sklearn.utils._param_validation import StrOptions
from ..tree._semisupervised_classes import (
    ExtraTreeClassifierSS,
    DecisionTreeClassifierSS,
    DecisionTreeRegressorSS,
    ExtraTreeRegressorSS,
    BipartiteDecisionTreeRegressorSS,
    BipartiteExtraTreeRegressorSS,
)
from ._forest import BaseMultipartiteForest

__all__ = [
    "ExtraTreesRegressorSS",
    "RandomForestRegressorSS",
    "BipartiteRandomForestRegressorSS",
    "BipartiteExtraTreesRegressorSS",
]


class BaseSemisupervisedForest(
    BaseForest,
    metaclass=ABCMeta,
):
    def fit(self, X, y, sample_weight=None, _X_targets=None):
        # TODO: Since we cannot pass _X_targets to trees in an ensemble,
        # inside sklearn.ensemble._forest._parallel_build_trees(),
        # we add _X_targets as a parameter in the forest.estimator_params
        # list, that will be reused by the individual trees without copying
        # the array for each tree and consuming a large amount of memory.
        # In forests, check_input is set to false, in which case we do not
        # override self._X_targets below.
        self._X_targets = self._check_X_targets(X, _X_targets)

        return super().fit(X, y, sample_weight)

    def _check_X_targets(self, X, _X_targets=None):
        self._validate_estimator()
        self._X_targets = None
        tree = self._make_estimator(append=False)
        return tree._check_X_targets(X, _X_targets)


class RandomForestClassifierSS(
    BaseSemisupervisedForest,
    ForestClassifier,
):

    _parameter_constraints: dict = {
        **ForestClassifier._parameter_constraints,
        **DecisionTreeClassifierSS._parameter_constraints,
        "class_weight": [
            StrOptions({"balanced_subsample", "balanced"}),
            dict,
            list,
            None,
        ],
    }

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        # Semi-supervised parameters:
        unsupervised_criterion="squared_error",
        supervision=0.5,
        update_supervision=None,
        ss_adapter=None,
        preprocess_X_targets=None,
    ):
        super().__init__(
            estimator=DecisionTreeClassifierSS(),
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
                # Semi-supervised parameters:
                "_X_targets",
                "supervision",
                "unsupervised_criterion",
                "update_supervision",
                "ss_adapter",
                "preprocess_X_targets",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

        # Semi-supervised parameters:
        self.unsupervised_criterion = unsupervised_criterion
        self.supervision = supervision
        self.update_supervision = update_supervision
        self.ss_adapter = ss_adapter
        self.preprocess_X_targets = preprocess_X_targets


class ExtraTreesClassifierSS(
    BaseSemisupervisedForest,
    ForestClassifier,
):
    _parameter_constraints: dict = {
        **ForestClassifier._parameter_constraints,
        **ExtraTreeClassifierSS._parameter_constraints,
        "class_weight": [
            StrOptions({"balanced_subsample", "balanced"}),
            dict,
            list,
            None,
        ],
    }

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        # Semi-supervised parameters:
        unsupervised_criterion="squared_error",
        supervision=0.5,
        update_supervision=None,
        ss_adapter=None,
        preprocess_X_targets=None,
    ):
        super().__init__(
            estimator=ExtraTreeClassifierSS(),
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
                # Semi-supervised parameters:
                "supervision",
                "unsupervised_criterion",
                "update_supervision",
                "ss_adapter",
                "preprocess_X_targets",
                "_X_targets",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

        # Semi-supervised parameters:
        self.unsupervised_criterion = unsupervised_criterion
        self.supervision = supervision
        self.update_supervision = update_supervision
        self.ss_adapter = ss_adapter
        self.preprocess_X_targets = preprocess_X_targets


class RandomForestRegressorSS(
    BaseSemisupervisedForest,
    ForestRegressor,
):
    _parameter_constraints: dict = {
        **ForestRegressor._parameter_constraints,
        **DecisionTreeRegressorSS._parameter_constraints,
    }

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        # Semi-supervised parameters:
        unsupervised_criterion="squared_error",
        supervision=0.5,
        update_supervision=None,
        ss_adapter=None,
        preprocess_X_targets=None,
    ):
        super().__init__(
            estimator=DecisionTreeRegressorSS(),
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
                # Semi-supervised parameters:
                "supervision",
                "unsupervised_criterion",
                "update_supervision",
                "ss_adapter",
                "preprocess_X_targets",
                "_X_targets",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

        # Semi-supervised parameters:
        self.unsupervised_criterion = unsupervised_criterion
        self.supervision = supervision
        self.update_supervision = update_supervision
        self.ss_adapter = ss_adapter
        self.preprocess_X_targets = preprocess_X_targets


class ExtraTreesRegressorSS(
    BaseSemisupervisedForest,
    ForestRegressor,
):
    _parameter_constraints: dict = {
        **ForestRegressor._parameter_constraints,
        **ExtraTreeRegressorSS._parameter_constraints,
    }

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        # Semi-supervised parameters:
        unsupervised_criterion="squared_error",
        supervision=0.5,
        update_supervision=None,
        ss_adapter=None,
        preprocess_X_targets=None,
    ):
        super().__init__(
            estimator=ExtraTreeRegressorSS(),
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
                # Semi-supervised parameters:
                "supervision",
                "unsupervised_criterion",
                "update_supervision",
                "ss_adapter",
                "preprocess_X_targets",
                "_X_targets",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

        # Semi-supervised parameters:
        self.unsupervised_criterion = unsupervised_criterion
        self.supervision = supervision
        self.update_supervision = update_supervision
        self.ss_adapter = ss_adapter
        self.preprocess_X_targets = preprocess_X_targets


class BaseSemisupervisedBipartiteForest(
    BaseMultipartiteForest,
    metaclass=ABCMeta,
):
    def fit(self, X, y, sample_weight=None, check_input=True, _X_targets=None):
        if check_input:
            # _X_targets will be reused by the individual trees without copying
            # the array in each of them and consuming a large amount of memory.
            _X_targets = self._check_X_targets(X, _X_targets)

        self._X_targets = _X_targets  # TODO: This is not necessary. Keep it?

        return super().fit(
            X, y,
            sample_weight=sample_weight,
            tree_fit_params={'_X_targets': _X_targets},
        )

    def _check_X_targets(self, X, _X_targets=None):
        self._validate_estimator()
        tree = self._make_estimator(append=False)
        return tree._check_X_targets(X, _X_targets)

    def _more_tags(self):
        self._validate_estimator()
        tree = self._make_estimator(append=False)
        return tree._more_tags()


class BipartiteRandomForestRegressorSS(
    BaseSemisupervisedBipartiteForest,
    ForestRegressor,
):

    _parameter_constraints: dict = {
        **BaseSemisupervisedBipartiteForest._parameter_constraints,
        **BipartiteDecisionTreeRegressorSS._parameter_constraints,
    }

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        # Semi-supervised parameters:
        supervision=0.5,
        ss_adapter=None,
        update_supervision=None,
        preprocess_X_targets=None,
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
        # Biaprtite and semisupervised parameters:
        unsupervised_criterion_rows="squared_error",
        unsupervised_criterion_cols="squared_error",
        axis_decision_only=False,
    ):
        super().__init__(
            estimator=BipartiteDecisionTreeRegressorSS(),
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
                # Semi-supervised parameters:
                "supervision",
                "unsupervised_criterion_rows",
                "unsupervised_criterion_cols",
                "update_supervision",
                "ss_adapter",
                "axis_decision_only",
                "preprocess_X_targets",
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
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

        # Semi-supervised parameters:
        self.supervision = supervision
        self.ss_adapter = ss_adapter
        self.unsupervised_criterion_rows = unsupervised_criterion_rows
        self.unsupervised_criterion_cols = unsupervised_criterion_cols
        self.update_supervision = update_supervision
        self.axis_decision_only = axis_decision_only
        self.preprocess_X_targets = preprocess_X_targets

        # Bipartite parameters:
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


class BipartiteExtraTreesRegressorSS(
    BaseSemisupervisedBipartiteForest,
    ForestRegressor,
):

    _parameter_constraints: dict = {
        **BaseSemisupervisedBipartiteForest._parameter_constraints,
        **BipartiteExtraTreeRegressorSS._parameter_constraints,
    }

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        # Semi-supervised parameters:
        supervision=0.5,
        ss_adapter=None,
        unsupervised_criterion_rows="squared_error",
        unsupervised_criterion_cols="squared_error",
        update_supervision=None,
        axis_decision_only=False,
        prediction_weights=None,
        preprocess_X_targets=None,
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
    ):
        super().__init__(
            estimator=BipartiteExtraTreeRegressorSS(),
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
                # Semi-supervised parameters:
                "supervision",
                "unsupervised_criterion_rows",
                "unsupervised_criterion_cols",
                "update_supervision",
                "ss_adapter",
                "axis_decision_only",
                "preprocess_X_targets",
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
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

        # Semi-supervised parameters:
        self.supervision = supervision
        self.ss_adapter = ss_adapter
        self.unsupervised_criterion_rows = unsupervised_criterion_rows
        self.unsupervised_criterion_cols = unsupervised_criterion_cols
        self.update_supervision = update_supervision
        self.axis_decision_only = axis_decision_only
        self.preprocess_X_targets = preprocess_X_targets

        # Bipartite parameters:
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
