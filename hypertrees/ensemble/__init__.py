"""
The :mod:`sklearn.ensemble` module includes ensemble-based methods for
classification, regression and anomaly detection.
"""
from ._forest import (
    RandomForestRegressor2D,
    ExtraTreesRegressor2D,
    BiclusteringRandomForestRegressor,
    BiclusteringExtraTreesRegressor,
)

__all__ = [
    "RandomForestRegressor2D"
    "ExtraTreesRegressor2D",
    "BiclusteringRandomForestRegressor",
    "BiclusteringExtraTreesRegressor",
]
