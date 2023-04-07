"""
The :mod:`sklearn.ensemble` module includes ensemble-based methods for
classification, regression and anomaly detection.
"""
from ._forest import (
    BipartiteRandomForestRegressor,
    BipartiteExtraTreesRegressor,
)
from ._semisupervised_forest import *
from ._gb import *

__all__ = [
    "BipartiteRandomForestRegressor"
    "BipartiteExtraTreesRegressor",
]
