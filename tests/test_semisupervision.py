from test_nd_classes import compare_trees, parse_args

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._splitter import BestSplitter

from hypertrees.tree import DecisionTreeRegressor2D
from hypertrees.tree._nd_splitter import make_2d_splitter
from hypertrees.tree._semisupervised_criterion import (
    SSMSE, SSCompositeCriterion, make_2dss_splitter, DynamicSSMSE,
)
from hypertrees.tree._semisupervised_classes import (
    DecisionTreeRegressorSS, DecisionTreeRegressor2DSS,
    DecisionTreeRegressorDS, DecisionTreeRegressor2DDS,
)

from sklearn.tree._criterion import MSE

import numpy as np
#from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64

from pathlib import Path
from time import time

import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# Default test params
DEF_PARAMS = dict(
    seed=7,
    shape=(50, 60),
    nattrs=(10, 9),
    nrules=10,
    min_samples_leaf=100,
    transpose_test=False,
    noise=0.1,
    inspect=False,
    plot=False,
    save_trees=False,
)


def test_supervised_component(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS

    treess = DecisionTreeRegressorSS(
        supervision=1,
        min_samples_leaf=PARAMS['min_samples_leaf'],
        random_state=PARAMS['seed'],
    )

    return compare_trees(
        tree1=treess,
        tree2=DecisionTreeRegressor,
        tree2_is_2d=False,
        **PARAMS,
    )


def test_unsupervised_component(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS

    treess = DecisionTreeRegressorSS(
        supervision=0,
        min_samples_leaf=PARAMS['min_samples_leaf'],
        random_state=PARAMS['seed'],
    )
    return compare_trees(
        tree1=DecisionTreeRegressor,
        tree2=treess,
        tree2_is_2d=False,
        tree1_is_unsupervised=True,
        **PARAMS,
    )


def test_supervised_component_2d(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS

    treess = DecisionTreeRegressor2DSS(
        supervision=1.,
        min_samples_leaf=PARAMS['min_samples_leaf'],
        random_state=PARAMS['seed'],
    )
    return compare_trees(
        tree1=DecisionTreeRegressor,
        tree1_is_unsupervised=False,
        tree2=treess,
        tree2_is_2d=True,
        **PARAMS,
    )


def test_unsupervised_component_2d(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS

    treess = DecisionTreeRegressor2DSS(
        supervision=0.,
        min_samples_leaf=PARAMS['min_samples_leaf'],
        random_state=PARAMS['seed'],
    )
    return compare_trees(
        tree1=DecisionTreeRegressor,
        tree1_is_unsupervised=True,
        tree2=treess,
        tree2_is_2d=True,
        **PARAMS,
    )


def test_semisupervision_1d2d(supervision=None, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    if supervision is None:
        rng = np.random.default_rng(PARAMS['seed'])
        supervision = rng.random()
    print('Supervision level:', supervision)

    tree1 = DecisionTreeRegressorSS(
        supervision=supervision,
        min_samples_leaf=PARAMS['min_samples_leaf'],
        random_state=PARAMS['seed'],
    )
    tree2 = DecisionTreeRegressor2DSS(
        supervision=supervision,
        min_samples_leaf=PARAMS['min_samples_leaf'],
        random_state=PARAMS['seed'],
    )

    return compare_trees(
        tree1=tree1,
        tree1_is_unsupervised=False,
        tree2=tree2,
        tree2_is_2d=True,
        **PARAMS,
    )


def test_dynamic_supervision_1d(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS

    # FIXME: Are not they supposed to match?
    return compare_trees(
        tree1=DecisionTreeRegressorDS,
        tree2=DecisionTreeRegressor2DDS,
        tree1_is_unsupervised=False,
        tree2_is_2d=True,
        **PARAMS,
    )


def main(**PARAMS):
    # test_supervised_component(**PARAMS)
    # test_unsupervised_component(**PARAMS)
    # test_supervised_component_2d(**PARAMS)
    # test_unsupervised_component_2d(**PARAMS)
    # test_semisupervision_1d2d(**PARAMS)
    test_dynamic_supervision_1d(**PARAMS)


if __name__ == "__main__":
    args = parse_args(**DEF_PARAMS)
    main(**vars(args))
