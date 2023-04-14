from pprint import pprint
import pytest
from sklearn.tree import (
    DecisionTreeClassifier, 
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from sklearn.ensemble import (
    RandomForestClassifier, 
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from bipartite_learn.tree import (
    BipartiteDecisionTreeRegressor,
    BipartiteExtraTreeRegressor,
)
from bipartite_learn.ensemble import (
    BipartiteRandomForestRegressor,
    BipartiteExtraTreesRegressor,
)
from bipartite_learn.tree._semisupervised_classes import (
    DecisionTreeClassifierSS, 
    DecisionTreeRegressorSS,
    ExtraTreeClassifierSS,
    ExtraTreeRegressorSS,
    BipartiteDecisionTreeRegressorSS,
    BipartiteExtraTreeRegressorSS,
)
from bipartite_learn.ensemble._semisupervised_forest import (
    RandomForestClassifierSS, 
    RandomForestRegressorSS,
    ExtraTreesClassifierSS,
    ExtraTreesRegressorSS,
    BipartiteRandomForestRegressorSS,
    BipartiteExtraTreesRegressorSS,
)
from .utils.test_utils import assert_equal_dicts

@pytest.mark.parametrize(
    'estimator1,estimator2,ignore',
    [
        # Semi-supervised
        (DecisionTreeClassifier, DecisionTreeClassifierSS, None),
        (DecisionTreeRegressor, DecisionTreeRegressorSS, None),
        (ExtraTreeClassifier, ExtraTreeClassifierSS, None),
        (ExtraTreeRegressor, ExtraTreeRegressorSS, None),

        # Semi-supervised forests
        (RandomForestClassifier, RandomForestClassifierSS, None),
        (RandomForestRegressor, RandomForestRegressorSS, None),
        (ExtraTreesClassifier, ExtraTreesClassifierSS, None),
        (ExtraTreesRegressor, ExtraTreesRegressorSS, None),

        # Bipartite
        (DecisionTreeRegressor, BipartiteDecisionTreeRegressor, None),
        (ExtraTreeRegressor, BipartiteExtraTreeRegressor, None),

        # Bipartite forests
        (RandomForestRegressor, BipartiteRandomForestRegressor, None),
        (ExtraTreesRegressor, BipartiteExtraTreesRegressor, None),

        # Bipartite and semi-supervised
        (DecisionTreeRegressorSS, BipartiteDecisionTreeRegressorSS, None),
        (BipartiteDecisionTreeRegressor, BipartiteDecisionTreeRegressorSS, None),
        (ExtraTreeRegressorSS, BipartiteExtraTreeRegressorSS, None),
        (BipartiteExtraTreeRegressor, BipartiteExtraTreeRegressorSS, None),

        # Bipartite and semi-supervised forests
        (RandomForestRegressorSS, BipartiteRandomForestRegressorSS, None),
        (BipartiteRandomForestRegressor, BipartiteRandomForestRegressorSS, None),
        (ExtraTreesRegressorSS, BipartiteExtraTreesRegressorSS, None),
        (BipartiteExtraTreesRegressor, BipartiteExtraTreesRegressorSS, None),
    ]
)
def test_default_params_are_consistent(estimator1, estimator2, ignore):
    assert_equal_dicts(
        estimator1().get_params(),
        estimator2().get_params(),
        ignore=ignore,
    )
