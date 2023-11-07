import logging
from pathlib import Path
from time import time
import numpy as np
from functools import partial
import pytest

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._splitter import BestSplitter
from sklearn.utils import check_random_state
from bipartite_learn.base import BaseMultipartiteEstimator
from bipartite_learn.tree import BipartiteDecisionTreeRegressor
from bipartite_learn.tree._splitter_factory import (
    make_bipartite_ss_splitter,
    make_semisupervised_criterion,
) 
from bipartite_learn.tree._semisupervised_classes import (
    DecisionTreeRegressorSS, BipartiteDecisionTreeRegressorSS,
)

from .test_bipartite_trees import compare_trees
from .utils.make_examples import make_interaction_regression, make_interaction_blobs
from .utils.test_utils import assert_equal_dicts

# from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

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

@pytest.fixture
def n_samples():
    return (50, 60)


@pytest.fixture
def n_features():
    return (10, 9)


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


@pytest.fixture
def regression_data(n_samples, n_features, random_state):
    X, Y, x, y = make_interaction_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=None,
        min_target=0.0,
        max_target=1.0,
        noise=0.0,
        return_molten=True,
        return_tree=False,
        random_state=random_state,
        max_depth=None,
    )
    return X, Y, x, y


@pytest.fixture
def classification_data(regression_data):
    X, Y, x, y = regression_data
    Y = (Y > Y.mean()).astype("float64")
    y = (y > y.mean()).astype("float64")
    return X, Y, x, y


@pytest.mark.parametrize(
    "tree",
    [
        DecisionTreeRegressor(),
        DecisionTreeRegressorSS(),
        BipartiteDecisionTreeRegressorSS(),
    ],
    ids=["sup", "ss_mono", "ss_bi"],
)
def test_redundant_splits(
    tree,
    supervision,
    random_state,
    classification_data,
):
    X, Y, x, y = classification_data

    tree.set_params(random_state=random_state)
    if not isinstance(tree, DecisionTreeRegressor):
        tree.set_params(supervision=supervision)

    if isinstance(tree, BaseMultipartiteEstimator):
        tree = tree.fit, X, Y
    else:
        tree = tree.fit, x, y

    tree_ = tree.tree_
    impurity = tree_.impurity
    single_label_remaining = np.ravel((tree_.value == 0) | (tree_.value == 1))

    # Assert all homogeneous nodes have impurity == 0.
    assert np.allclose(
        impurity[single_label_remaining], 0
    ), "Some homogeneous nodes have impurity != 0."

    # Assert all nodes with impurity == 0 are leaves.
    assert np.all(
        tree_.children_left[impurity == 0]
        == tree_.children_right[impurity == 0]
    ), "Some nodes with impurity == 0 are not leaves."

    # Assert all homogeneous nodes are leaves.
    assert np.all(
        tree_.children_left[single_label_remaining]
        == tree_.children_right[single_label_remaining]
    ), "Some homogeneous nodes are not leaves."


def test_monopartite_semisupervised(
    supervision,
    msl,
    random_state,
    n_samples,
    n_features,
):
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
        n_samples=n_samples,
        n_features=n_features,
        min_target=0.0,
        max_target=100.0,
        noise=0.1,
    )


def test_semisupervision_1d2d(
    supervision,
    random_state,
    msl,
    max_depth,
    n_samples,
    n_features,
):
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
        n_samples=n_samples,
        n_features=n_features,
        min_target=0.0,
        max_target=100.0,
        noise=0.1,
    )


@pytest.mark.parametrize('update_supervision', [
    lambda **_: 0.7,
    lambda original_supervision, **_: original_supervision,
    lambda weighted_n_node_samples, weighted_n_samples, **_:
        1 - 0.3 * weighted_n_node_samples/weighted_n_samples,
])
def test_dynamic_supervision_1d2d(
    supervision,
    random_state,
    msl,
    max_depth,
    update_supervision,
    n_samples,
    n_features,
):
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
        n_samples=n_samples,
        n_features=n_features,
        min_target=0.0,
        max_target=100.0,
        noise=0.1,
    )
