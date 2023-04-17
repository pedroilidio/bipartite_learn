"""
The :mod:`sklearn.ensemble` module includes ensemble-based methods for
classification, regression and anomaly detection.
"""
from ._forest import (
    BipartiteRandomForestRegressor,
    BipartiteExtraTreesRegressor,
)
from ._semisupervised_forest import (
    RandomForestRegressorSS,
    ExtraTreesRegressorSS,
    BipartiteRandomForestRegressorSS,
    BipartiteExtraTreesRegressorSS,
)
from ._gb import (
    BipartiteGradientBoostingRegressor,
    BipartiteGradientBoostingClassifier,
)

__all__ = [
    "BipartiteRandomForestRegressor",
    "BipartiteExtraTreesRegressor",
    "BipartiteGradientBoostingRegressor",
    "BipartiteGradientBoostingClassifier",
    "RandomForestRegressorSS",
    "ExtraTreesRegressorSS",
    "BipartiteRandomForestRegressorSS",
    "BipartiteExtraTreesRegressorSS",
]
