"""
The :mod:`sklearn.ensemble` module includes ensemble-based methods for
classification, regression and anomaly detection.
"""
from ._forest import (
    BipartiteRandomForestRegressor,
    BipartiteExtraTreesRegressor,
)

__all__ = [
    "BipartiteRandomForestRegressor"
    "BipartiteExtraTreesRegressor",
]
