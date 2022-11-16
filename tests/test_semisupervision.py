import logging
from pathlib import Path
from time import time
import numpy as np
from functools import partial
import pytest

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._splitter import BestSplitter
from sklearn.utils import check_random_state

from hypertrees.tree import BipartiteDecisionTreeRegressor
from hypertrees.tree._splitter_factory import (
    make_2dss_splitter,
)
from hypertrees.tree._semisupervised_criterion import (
    SSMSE, SSCompositeCriterion,
    SingleFeatureSSCompositeCriterion, MSE2DSFSS,
)

from hypertrees.tree._semisupervised_classes import (
    DecisionTreeRegressorSS, DecisionTreeRegressor2DSS,
)

from hypertrees.tree._semisupervised_splitter import BestSplitterSFSS
from hypertrees.tree._dynamic_supervision_criterion import DynamicSSMSE

from sklearn.tree._criterion import MSE
from test_nd_classes import compare_trees, parse_args

#from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# Default test params
DEF_PARAMS = dict(
    n_samples=(50, 60),
    n_features=(10, 9),
    min_samples_leaf=100,
    noise=0.1,
    supervision=-1.,
    random_state=0,
)


def test_supervised_component(**params):
    params = DEF_PARAMS | params

    treess = DecisionTreeRegressorSS(
        supervision=1,
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
    )

    return compare_trees(
        tree1=treess,
        tree2=DecisionTreeRegressor,
        tree2_is_2d=False,
        **params,
    )


def test_unsupervised_component(**params):
    params = DEF_PARAMS | params

    treess = DecisionTreeRegressorSS(
        supervision=0,
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
    )
    return compare_trees(
        tree1=DecisionTreeRegressor,
        tree2=treess,
        tree2_is_2d=False,
        tree1_is_unsupervised=True,
        **params,
    )


def test_supervised_component_2d(**params):
    params = DEF_PARAMS | params

    treess = DecisionTreeRegressor2DSS(
        supervision=1.,
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
    )
    return compare_trees(
        tree1=DecisionTreeRegressor,
        tree1_is_unsupervised=False,
        tree2=treess,
        tree2_is_2d=True,
        **params,
    )


def test_unsupervised_component_2d(**params):
    params = DEF_PARAMS | params

    treess = DecisionTreeRegressor2DSS(
        supervision=0.,
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
    )
    return compare_trees(
        tree1=DecisionTreeRegressor,
        tree1_is_unsupervised=True,
        tree2=treess,
        tree2_is_2d=True,
        **params,
    )


def test_semisupervision_1d2d(supervision=None, **params):
    params = DEF_PARAMS | params
    rstate = check_random_state(params['random_state'])
    if supervision in (None, -1):
        supervision = rstate.random()
    print('Supervision level:', supervision)

    tree1 = DecisionTreeRegressorSS(
        supervision=supervision,
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
    )
    tree2 = DecisionTreeRegressor2DSS(
        supervision=supervision,
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
    )

    breakpoint()
    return compare_trees(
        tree1=tree1,
        tree1_is_unsupervised=False,
        tree2=tree2,
        tree2_is_2d=True,
        **params,
    )


@pytest.mark.skip
def test_dynamic_supervision_1d2d(**params):
    params = DEF_PARAMS | params

    # FIXME: Are not they supposed to match?
    return compare_trees(
        tree1=DecisionTreeRegressorDS,
        tree2=DecisionTreeRegressor2DDS,
        tree1_is_unsupervised=False,
        tree2_is_2d=True,
        **params,
    )


@pytest.mark.skip
def test_single_feature_semisupervision_1d_sup(**params):
    params = DEF_PARAMS | params
    rstate = check_random_state(params['random_state'])

    splitter1d = BestSplitterSFSS(
        criterion=SingleFeatureSSCompositeCriterion(
            supervision=1.,
            criterion=MSE,
            n_features=np.sum(params['n_features']),
            n_samples=np.prod(params['n_samples']),
            n_outputs=1,
        ),
        max_features=np.sum(params['n_features']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=rstate,
    )

    tree1 = DecisionTreeRegressorSS(
        splitter=splitter1d,
    )

    return compare_trees(
        tree1=tree1,
        tree2_is_2d=True,
        **params,
    )


@pytest.mark.skip
def test_single_feature_semisupervision_1d2d(supervision=None, **params):
    params = DEF_PARAMS | params
    rstate = check_random_state(params['random_state'])
    if supervision is None:
        supervision = rstate.random()
    print('Supervision level:', supervision)

    splitter1d = BestSplitterSFSS(
        criterion=SingleFeatureSSCompositeCriterion(
            supervision=supervision,
            criterion=MSE,
            n_features=1.,
            n_samples=np.prod(params['n_samples']),
            n_outputs=1,
        ),
        max_features=np.sum(params['n_features']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=rstate,
    )

    ss2d_splitter = make_2dss_splitter(
        splitters=BestSplitterSFSS,
        criteria=MSE,
        ss_criteria=SingleFeatureSSCompositeCriterion,
        supervision=supervision,
        max_features=params['n_features'],
        n_features=1,
        n_samples=params['n_samples'],
        n_outputs=1,
        random_state=rstate,
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
    )

    tree1 = DecisionTreeRegressorSS(
        splitter=splitter1d,
    )
    tree2 = DecisionTreeRegressor2DSS(
        splitter=ss2d_splitter,
    )

    return compare_trees(
        tree1=tree1,
        tree2=tree2,
        tree1_is_unsupervised=False,
        tree2_is_2d=True,
        **params,
    )


@pytest.mark.skip
def test_single_feature_semisupervision_1d2d_classes(**params):
    params = DEF_PARAMS | params
    rstate = check_random_state(params['random_state'])
    supervision = params['supervision']

    if supervision == -1:
        supervision = rstate.random()

    print('Supervision level:', supervision)

    # FIXME: Are not they supposed to match?
    return compare_trees(
        tree1=partial(DecisionTreeRegressorSFSS, supervision=supervision),
        tree2=partial(DecisionTreeRegressor2DSFSS, supervision=supervision),
        tree1_is_unsupervised=False,
        tree2_is_2d=True,
        **params,
    )


def main(**params):
    params = DEF_PARAMS | params
    test_supervised_component(**params)
    test_unsupervised_component(**params)
    test_supervised_component_2d(**params)
    test_unsupervised_component_2d(**params)

    # FIXME: random_state=82; random_state=3 nrules=3
    # FIXME: --random_state 2133 --supervision .03
    # FIXME: --random_state 82 --supervision .5
    # When actual impurity is used intead of the proxies
    # --random_state 8221324 --supervision .3
    # --random_state 31284009 --supervision .3
    # --random_state 1 --supervision .1
    # --random_state 2 --supervision .1
    test_semisupervision_1d2d(**params)

    test_dynamic_supervision_1d2d(**params)
    # test_single_feature_semisupervision_1d_sup(**params)
    # test_single_feature_semisupervision_1d2d(**params)  # FIXME
    # test_single_feature_semisupervision_1d2d_classes(**params)


if __name__ == "__main__":
    args = parse_args(**DEF_PARAMS)
    params = vars(args)

    if params['seed_end'] == -1:
        main(**params)
    else:
        unsuccessful = []
        nseeds = 100

        for s in range(params['random_state'], params['seed_end']):
            params['random_state'] = s
            try:
                main(**params)
            except AssertionError:
                unsuccessful.append(s)

        print(
            f'Success rate: {len(unsuccessful)}/{nseeds} = {100*len(unsuccessful)/nseeds:.3f}%')
        print('Failed seeds:', *unsuccessful)
