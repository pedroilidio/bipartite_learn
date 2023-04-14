import logging
from pathlib import Path
from time import time
import numpy as np
from functools import partial
import pytest

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._splitter import BestSplitter
from sklearn.utils import check_random_state

from bipartite_learn.tree import BipartiteDecisionTreeRegressor
from bipartite_learn.tree._splitter_factory import (
    make_bipartite_ss_splitter,
    make_semisupervised_criterion,
) 
from bipartite_learn.tree._semisupervised_criterion import (
    SSCompositeCriterion,
)

from bipartite_learn.tree._semisupervised_classes import (
    DecisionTreeRegressorSS, BipartiteDecisionTreeRegressorSS,
)

from sklearn.tree._criterion import MSE
from .test_bipartite_trees import compare_trees, parse_args

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
    return request.param


# @pytest.fixture(params=[None, 5])
@pytest.fixture(params=[1, 2, 3])
def max_depth(request):
    return request.param


# NOTE: small leaves may lead to differing trees by chance. Simply becase,
# with just a few samples, one feature column may ocasionally have values with
# the same order as another column, so that they yield the same split position
# and impurities. Since we cannot dictate the order in which features are
# chosen to be evaluated by the splitters, the two equally-ordered features
# can be swapped.
# @pytest.fixture(params=[1, 6, 10, .1])
@pytest.fixture(params=[0.05])
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
    tree2 = BipartiteDecisionTreeRegressorSS(
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


@pytest.mark.parametrize('update_supervision', [
    lambda **_: 0.7,
    lambda original_supervision, **_: original_supervision,
    lambda weighted_n_node_samples, weighted_n_samples, **_:
        1 - 0.3 * weighted_n_node_samples/weighted_n_samples,
])
def test_dynamic_supervision_1d2d(
    supervision, random_state, msl, max_depth, update_supervision, **params,
):
    params = DEF_PARAMS | params
    mono_sup_values = []
    bi_sup_values = []

    def mono_update_supervision(**kwargs):
        mono_sup_values.append(kwargs['current_supervision'])
        return update_supervision(**kwargs)
    def bi_update_supervision(**kwargs):
        bi_sup_values.append(kwargs['current_supervision'])
        return update_supervision(**kwargs)

    tree1 = DecisionTreeRegressorSS(
        criterion="squared_error",
        unsupervised_criterion="squared_error",
        supervision=supervision,
        update_supervision=mono_update_supervision,
        min_samples_leaf=msl,
        max_depth=max_depth,
    )
    tree2 = BipartiteDecisionTreeRegressorSS(
        criterion="squared_error",
        unsupervised_criterion_rows="squared_error",
        unsupervised_criterion_cols="squared_error",
        supervision=supervision,
        update_supervision=bi_update_supervision,
        min_samples_leaf=msl,
        max_depth=max_depth,
    )

    return compare_trees(
        tree1=tree1,
        tree2=tree2,
        tree2_is_2d=True,
        supervision=1.0,
        random_state=random_state,
        **params,
    )
