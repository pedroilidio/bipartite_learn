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
    # n_samples=(150, 160),
    n_samples=(50, 60),
    n_features=(10, 9),
    # n_targets=(2, 1),
    min_target=0.0,
    max_target=100.0,
    # NOTE: setting noise=0 makes some comparisons with supervision=1.0 fail.
    # I suppose it can be due to two feature columns occasionally having the
    # same order, especially when there are few samples in the node.
    noise=0.1,  
)


# FIXME: very small values sometimes cause trees to differ, probably due to
# precision issues.
@pytest.fixture(params=[0.0, 0.0001, 0.1, 0.2328, 0.569, 0.782, 0.995, 1.0])
def supervision(request):
    return request.param


@pytest.fixture(params=range(3))
def random_state(request):
    # return request.param
    return 0


@pytest.fixture(params=[None, 5])
def max_depth(request):
    return request.param


@pytest.fixture(params=[2, 5, None])
def gen_tree_max_depth(request):
    """max depth of dataset's generator random tree"""
    return request.param


# NOTE: small leaves may lead to differing trees by chance. Simply becase,
# with just a few samples, one feature column may ocasionally have values with
# the same order as another column, so that they yield the same split position
# and impurities. Since we cannot dictate the order in which features are
# chosen to be evaluated by the splitters, the two equally-ordered features
# can be swapped.
@pytest.fixture(params=[1, 6, 10, .1])
def msl(request):  # min_samples_leaf parameter
    return request.param


def test_monopartite_semisupervised(supervision, msl, random_state, **params):
    params = DEF_PARAMS | params

    tree_ss = DecisionTreeRegressorSS(
        supervision=supervision,
        min_samples_leaf=msl,
        random_state=random_state,
    )
    tree1 = DecisionTreeRegressor(
        min_samples_leaf=msl,
        random_state=random_state,
    )

    return compare_trees(
        tree1=tree1,
        tree2=tree_ss,
        tree2_is_2d=False,
        supervision=supervision,
        random_state=random_state,
        **params,
    )


def test_semisupervision_1d2d(
    supervision,
    random_state,
    msl,
    max_depth,
    **params,
):
    params = DEF_PARAMS | params
    # n_samples = np.prod(params['n_samples'])
    # max_depth = int(np.ceil(np.log2(n_samples/6))) if msl == 1 else None

    print('* Supervision level:', supervision)
    print('* Max depth:', max_depth)

    tree1 = DecisionTreeRegressorSS(
        supervision=supervision,
        min_samples_leaf=msl,
        random_state=random_state,
        max_depth=max_depth,
    )
    tree2 = DecisionTreeRegressor2DSS(
        supervision=supervision,
        min_samples_leaf=msl,
        random_state=random_state,
        max_depth=max_depth,
    )

    return compare_trees(
        tree1=tree1,
        tree2=tree2,
        tree2_is_2d=True,
        supervision=1.0,  # tree1 will already apply the supervision
        random_state=random_state,
        **params,
    )


# FIXME: supervision=0.0 fails. No idea why, it shoul not even be used.
# @pytest.mark.skip(
#     reason="compare_trees still does not work with dynamic supervision")
@pytest.mark.parametrize('update_supervision', [
    lambda weighted_n_node_samples, weighted_n_samples, **kw: 0.7,
    lambda weighted_n_node_samples, weighted_n_samples, **kw:
        1 - 0.6 * weighted_n_node_samples/weighted_n_samples,
])
def test_dynamic_supervision_1d2d(
    supervision, update_supervision, msl, random_state, **params,
):
    params = DEF_PARAMS | params

    # def new_update_supervision(**kwargs):
    #     print(kwargs)
    #     return update_supervision(**kwargs)

    # def new_update_supervision(**kwargs):
    #     return supervision

    tree1 = DecisionTreeRegressorSS(
        criterion="squared_error",
        unsupervised_criterion="squared_error",
        supervision=supervision,
        update_supervision=update_supervision,
        min_samples_leaf=msl,
        # max_depth=max_depth,
    )
    tree2 = DecisionTreeRegressor2DSS(
        criterion="squared_error",
        unsupervised_criterion_rows="squared_error",
        unsupervised_criterion_cols="squared_error",
        supervision=supervision,
        update_supervision=update_supervision,
        min_samples_leaf=msl,
        # max_depth=max_depth,
    )

    return compare_trees(
        tree1=tree1,
        tree2=tree2,
        tree2_is_2d=True,
        supervision=1.0,
        random_state=random_state,
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
        min_weight_leaf=0.0,
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
        random_state=random_state,
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
