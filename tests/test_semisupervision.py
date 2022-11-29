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
    make_semisupervised_criterion,
) 
from hypertrees.tree._semisupervised_criterion import (
    SSCompositeCriterion,
    SingleFeatureSSCompositeCriterion,
)

from hypertrees.tree._semisupervised_classes import (
    DecisionTreeRegressorSS, DecisionTreeRegressor2DSS,
)

from hypertrees.tree._semisupervised_splitter import BestSplitterSFSS
from hypertrees.tree._dynamic_supervision_criterion import DynamicSSMSE

from sklearn.tree._criterion import MSE
from test_nd_classes import compare_trees, parse_args

# from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# Default test params
DEF_PARAMS = dict(
    n_samples=(150, 160),
    n_features=(10, 9),
    # n_targets=(2, 1),
    min_target=0.0,
    max_target=100.0,
    min_samples_leaf=100,
    noise=0.0,
    random_state=0,
)


@pytest.fixture(
    params=[0.0, 0.00001, 0.1, 0.2328, 0.569, 0.782, 0.995, 1.0])
def supervision(request):
    return request.param


@pytest.fixture(
    params=[2, 5, None])
def max_depth(request):
    """max depth of dataset's generator random tree"""
    return request.param


def test_monopartite_semisupervised(supervision, **params):
    params = DEF_PARAMS | params

    treess = DecisionTreeRegressorSS(
        supervision=supervision,
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
    )

    return compare_trees(
        tree1=DecisionTreeRegressor,
        tree2=treess,
        tree2_is_2d=False,
        supervision=supervision,
        **params,
    )


def test_supervised_component(**params):
    params = DEF_PARAMS | params

    treess = DecisionTreeRegressorSS(
        supervision=1.0,
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
    )

    return compare_trees(
        tree1=DecisionTreeRegressor,
        tree2=treess,
        tree2_is_2d=False,
        supervision=1.0,
        **params,
    )


@pytest.mark.parametrize('random_state', range(10))
def test_unsupervised_component(random_state, max_depth, **params):
    params = DEF_PARAMS | params
    params['random_state'] = random_state
    params['max_depth'] = max_depth

    treess = DecisionTreeRegressorSS(
        supervision=0,
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
    )
    return compare_trees(
        tree1=DecisionTreeRegressor,
        tree2=treess,
        tree2_is_2d=False,
        supervision=0.0,
        **params,
    )


# FIXME: Fails for some seeds if make_interraction_data is used instead of
#        make_interaction_regression in test_nd_classes.py::compare_trees().
#        On the other hand, intermediate supervision values (not 0 or 1) fail
#        more often.
@pytest.mark.parametrize('random_state', range(10))
def test_supervised_component_2d(random_state, max_depth, **params):
    params = DEF_PARAMS | params
    params['random_state'] = random_state
    params['max_depth'] = max_depth

    treess = DecisionTreeRegressor2DSS(
        supervision=1.,
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state'],
    )
    return compare_trees(
        tree1=DecisionTreeRegressor,
        tree2=treess,
        tree2_is_2d=True,
        supervision=1.0,
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
        tree2=treess,
        tree2_is_2d=True,
        supervision=0.0,
        **params,
    )


def test_semisupervision_1d2d(supervision, max_depth, **params):
    params = DEF_PARAMS | params
    print('Supervision level:', supervision)
    params['noise'] = 0.0
    params['max_depth'] = max_depth

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

    return compare_trees(
        tree1=tree1,
        tree2=tree2,
        tree2_is_2d=True,
        supervision=1.0,  # tree1 will multiply the supervision
        **params,
    )


@pytest.mark.skip(
    reason="compare_trees still does not work with dynamic supervision")
@pytest.mark.parametrize('max_depth', [None, 1, 10])
def test_dynamic_supervision_1d2d(supervision, max_depth, **params):
    params = DEF_PARAMS | params

    def update_supervision(
        weighted_n_samples,
        weighted_n_node_samples,
        criterion,
    ):
        w = weighted_n_node_samples
        W = weighted_n_samples
        print(f'{w=} {W=}')
# XXX: Weights differ between 12 and 2D, and might be the cause of tree disparity
        return 1/(1 + 2 ** (10*(0.5 - np.log2(w)/np.log2(W))))

    tree1 = DecisionTreeRegressorSS(
        criterion="squared_error",
        unsupervised_criterion="squared_error",
        supervision=supervision,
        update_supervision=update_supervision,
        min_samples_leaf=1,
        max_depth=max_depth,
    )
    tree2 = DecisionTreeRegressor2DSS(
        criterion="squared_error",
        unsupervised_criterion_rows="squared_error",
        unsupervised_criterion_cols="squared_error",
        supervision=supervision,
        update_supervision=update_supervision,
        min_samples_leaf=1,
        max_depth=max_depth,
    )

    return compare_trees(
        tree1=tree1,
        tree2=tree2,
        tree2_is_2d=True,
        supervision=1.0,
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
    if supervision is None:
        supervision = check_random_state(params['random_state']).random()
    print('Supervision level:', supervision)

    splitter1d = BestSplitterSFSS(
        criterion=SingleFeatureSSCompositeCriterion(
            supervision=supervision,
            supervised_criterion=MSE,
            unsupervised_criterion=MSE,
            n_features=1.,
            n_samples=np.prod(params['n_samples']),
            n_outputs=1,
        ),
        max_features=np.sum(params['n_features']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=check_random_state(params['random_state']),
    )

    ss2d_splitter = make_2dss_splitter(
        splitters=BestSplitterSFSS,
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        ss_criteria=SingleFeatureSSCompositeCriterion,
        supervision=supervision,
        max_features=params['n_features'],
        n_features=1,
        n_samples=params['n_samples'],
        n_outputs=1,
        random_state=check_random_state(params['random_state']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
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
