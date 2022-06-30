# TODO: Documentation.
# TODO: Classifiers.

from sklearn.ensemble._forest import ForestRegressor

from ._forest import ForestRegressorND
from ..tree._nd_classes import ExtraTreeRegressor2D, DecisionTreeRegressor2D
from ..tree._semisupervised_classes import (
    DecisionTreeRegressorSS,
    ExtraTreeRegressorSS,
    DecisionTreeRegressor2DSS,
    ExtraTreeRegressor2DSS,
    DecisionTreeRegressorSFSS,
    ExtraTreeRegressorSFSS,
    DecisionTreeRegressor2DSFSS,
    ExtraTreeRegressor2DSFSS,
)

__all__ = [
    "ExtraTreesRegressorSS",
    "RandomForestRegressorSS",
    "RandomForestRegressor2DSS",
    "ExtraTreesRegressor2DSS",
]


class RandomForestRegressorSS(ForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,

        # Semi-supervised parameters:
        supervision=.5,
        criterion=None,
        ss_criterion="ss_squared_error",  # NOTE
        supervised_criterion=None,
        unsupervised_criterion=None,

        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=DecisionTreeRegressorSS(),  # NOTE
            n_estimators=n_estimators,
            estimator_params=(
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
                "ss_criterion",
                "criterion",
                "supervised_criterion",
                "unsupervised_criterion",
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
        self.ss_criterion = ss_criterion
        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion


class ExtraTreesRegressorSS(ForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,

        # Semi-supervised parameters:
        supervision=.5,
        ss_criterion="ss_squared_error",  # NOTE
        criterion=None,
        supervised_criterion=None,
        unsupervised_criterion=None,

        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=ExtraTreeRegressorSS(),  # NOTE
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
                "ss_criterion",
                "supervised_criterion",
                "unsupervised_criterion",
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
        self.ss_criterion = ss_criterion
        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion


class RandomForestRegressor2DSS(ForestRegressorND):
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,

        # 2D parameters:
        ax_min_samples_leaf=1,
        ax_min_weight_fraction_leaf=None,
        ax_max_features=None,
        criterion_wrapper="ss_squared_error",

        # Semi-supervised parameters:
        supervision=.5,
        ss_criterion="ss_squared_error",  # NOTE
        supervised_criterion=None,
        unsupervised_criterion=None,

        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=DecisionTreeRegressor2DSS(),  # NOTE
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

                # 2D parameters:
                "ax_min_samples_leaf",
                "ax_min_weight_fraction_leaf",
                "ax_max_features",
                "criterion_wrapper",

                # Semi-supervised parameters:
                "supervision",
                "ss_criterion",
                "supervised_criterion",
                "unsupervised_criterion",
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

        # 2D parameters:
        self.ax_min_samples_leaf = ax_min_samples_leaf
        self.ax_min_weight_fraction_leaf = ax_min_weight_fraction_leaf
        self.ax_max_features = ax_max_features
        self.criterion_wrapper = criterion_wrapper

        # Semi-supervised parameters:
        self.supervision = supervision
        self.ss_criterion = ss_criterion
        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion


class ExtraTreesRegressor2DSS(ForestRegressorND):
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,

        # 2D parameters:
        ax_min_samples_leaf=1,
        ax_min_weight_fraction_leaf=None,
        ax_max_features=None,
        criterion_wrapper="ss_squared_error",

        # Semi-supervised parameters:
        supervision=.5,
        ss_criterion="ss_squared_error",  # NOTE
        supervised_criterion=None,
        unsupervised_criterion=None,

        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=ExtraTreeRegressor2DSS(),  # NOTE
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

                # 2D parameters:
                "ax_min_samples_leaf",
                "ax_min_weight_fraction_leaf",
                "ax_max_features",
                "criterion_wrapper",

                # Semi-supervised parameters:
                "supervision",
                "ss_criterion",
                "supervised_criterion",
                "unsupervised_criterion",
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

        # 2D parameters:
        self.ax_min_samples_leaf = ax_min_samples_leaf
        self.ax_min_weight_fraction_leaf = ax_min_weight_fraction_leaf
        self.ax_max_features = ax_max_features
        self.criterion_wrapper = criterion_wrapper

        # Semi-supervised parameters:
        self.supervision = supervision
        self.ss_criterion = ss_criterion
        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion


# ============================================================================= 
# Single Feature Semi-supervised Forests
# ============================================================================= 


class RandomForestRegressorSFSS(ForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,

        # Semi-supervised parameters:
        supervision=.5,
        criterion="squared_error",
        ss_criterion="single_feature_ss",
        supervised_criterion=None,
        unsupervised_criterion=None,

        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=DecisionTreeRegressorSFSS(),  # NOTE
            n_estimators=n_estimators,
            estimator_params=(
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
                "ss_criterion",
                "criterion",
                "supervised_criterion",
                "unsupervised_criterion",
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
        self.ss_criterion = ss_criterion
        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion


class ExtraTreesRegressorSFSS(ForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,

        # Semi-supervised parameters:
        supervision=.5,
        ss_criterion="single_feature_ss",  # NOTE
        criterion="squared_error",
        supervised_criterion=None,
        unsupervised_criterion=None,

        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=ExtraTreeRegressorSFSS(),  # NOTE
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
                "ss_criterion",
                "supervised_criterion",
                "unsupervised_criterion",
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
        self.ss_criterion = ss_criterion
        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion


class RandomForestRegressor2DSFSS(ForestRegressorND):
    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,

        # 2D parameters:
        ax_min_samples_leaf=1,
        ax_min_weight_fraction_leaf=None,
        ax_max_features=None,
        criterion_wrapper="sfss_squared_error",

        # Semi-supervised parameters:
        supervision=.5,
        ss_criterion="single_feature_ss",  # NOTE
        criterion="squared_error",
        supervised_criterion=None,
        unsupervised_criterion=None,

        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=DecisionTreeRegressor2DSFSS(),  # NOTE
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

                # 2D parameters:
                "ax_min_samples_leaf",
                "ax_min_weight_fraction_leaf",
                "ax_max_features",
                "criterion_wrapper",

                # Semi-supervised parameters:
                "supervision",
                "ss_criterion",
                "supervised_criterion",
                "unsupervised_criterion",
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

        # 2D parameters:
        self.ax_min_samples_leaf = ax_min_samples_leaf
        self.ax_min_weight_fraction_leaf = ax_min_weight_fraction_leaf
        self.ax_max_features = ax_max_features
        self.criterion_wrapper = criterion_wrapper

        # Semi-supervised parameters:
        self.supervision = supervision
        self.ss_criterion = ss_criterion
        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion


class ExtraTreesRegressor2DSFSS(ForestRegressorND):
    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,

        # 2D parameters:
        ax_min_samples_leaf=1,
        ax_min_weight_fraction_leaf=None,
        ax_max_features=None,
        criterion_wrapper="sfss_squared_error",

        # Semi-supervised parameters:
        supervision=.5,
        ss_criterion="single_feature_ss",  # NOTE
        criterion="squared_error",
        supervised_criterion=None,
        unsupervised_criterion=None,

        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            base_estimator=ExtraTreeRegressor2DSFSS(),  # NOTE
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

                # 2D parameters:
                "ax_min_samples_leaf",
                "ax_min_weight_fraction_leaf",
                "ax_max_features",
                "criterion_wrapper",

                # Semi-supervised parameters:
                "supervision",
                "ss_criterion",
                "supervised_criterion",
                "unsupervised_criterion",
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

        # 2D parameters:
        self.ax_min_samples_leaf = ax_min_samples_leaf
        self.ax_min_weight_fraction_leaf = ax_min_weight_fraction_leaf
        self.ax_max_features = ax_max_features
        self.criterion_wrapper = criterion_wrapper

        # Semi-supervised parameters:
        self.supervision = supervision
        self.ss_criterion = ss_criterion
        self.supervised_criterion = supervised_criterion
        self.unsupervised_criterion = unsupervised_criterion
