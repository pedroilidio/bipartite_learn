import copy
import logging
import numpy as np
from itertools import product
from pprint import pprint
from typing import Callable

import pytest
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
import sklearn.tree
# from sklearn.tree._tree import Tree
from sklearn.utils._testing import assert_allclose
from sklearn.utils.validation import check_symmetric, check_random_state

from bipartite_learn.tree._bipartite_classes import (
    BipartiteDecisionTreeRegressor,
    BipartiteDecisionTreeClassifier,
    _normalize_weights,
)
from bipartite_learn.tree import _bipartite_classes
from bipartite_learn.melter import row_cartesian_product

from .utils.make_examples import make_interaction_blobs, make_interaction_regression, make_interaction_data
from .utils.test_utils import assert_equal_dicts, stopwatch, parse_args, gen_mock_data, melt_2d_data

# from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# Default test params
DEF_PARAMS = dict(
    n_samples=(51, 60),
    n_features=(10, 9),
    # noise=2,
    noise=0.1,
    # noise=0.0,  # makes it unable to reach 1 sample per leaf
)


@pytest.fixture(params=range(10))
def random_state(request):
    return request.param


@pytest.fixture(params=[(37, 61)])
def n_samples(request):
    return request.param


@pytest.fixture(params=[(10, 9)])
def n_features(request):
    return request.param

@pytest.fixture(params=["squared_error_gso", "friedman_gso"])
def gso_criterion(request):
    return request.param

def get_leaves(tree: sklearn.tree._tree.Tree):
    # the Tree object can be accessed from tree estimators with estimator.tree_
    ch_left = tree.children_left
    ch_right = tree.children_right
    leaf_mask = ch_right == ch_left  # == -1 == sklearn.tree._tree.TREE_LEAF

    return dict(
        value=tree.value[leaf_mask],
        impurity=tree.impurity[leaf_mask],
        n_node_samples=tree.n_node_samples[leaf_mask],
        weighted_n_node_samples=tree.weighted_n_node_samples[leaf_mask],
    )


def assert_equal_leaves(
    tree1, tree2, verbose=False, ignore=None, rtol=1e-7, atol=1e-8,
):
    n_leaves1 = tree1.n_leaves
    n_leaves2 = tree2.n_leaves
    assert n_leaves1 == n_leaves2, (
        f'Number of leaves differ: {n_leaves1} != {n_leaves2}.'
    )
    leaves1 = get_leaves(tree1)
    leaves2 = get_leaves(tree2)

    for leaves in (leaves1, leaves2):
        # It is normal for the last leaves to differ in their order.
        keys_to_order = [
            leaves['impurity'].reshape(-1),
            leaves['weighted_n_node_samples'].reshape(-1),
        ]
        if ignore is None or 'value' not in ignore:
            if leaves1['value'].shape != leaves2['value'].shape:
                raise ValueError(
                    'Diferent leaf value shapes, you are probably working with '
                    'mixed classification-regression trees. To ignore leaf '
                    'values include "value" in the ignore parameter.'
                )
            keys_to_order.append(leaves['value'].reshape(-1))

        order = np.lexsort(np.vstack(keys_to_order))

        for k, v in leaves.items():
            leaves[k] = v[order]
            # leaves[k] = np.sort(leaves[k].flatten())
            # leaves[k] = np.sort(np.unique(leaves[k].flatten()))

    assert_equal_dicts(leaves1, leaves2, ignore=ignore, rtol=rtol, atol=atol)


# TODO: parameter description.
def compare_trees(
    tree1,
    tree2,
    random_state,
    tree2_is_2d=True,
    supervision=1.0,
    max_gen_depth=None,
    verbose=False,
    rtol=1e-7,
    atol=1e-8,
    **params,
):
    """Fit and compare trees.

    Fit bipartite_learnBipartiteDecisionTreeRegressor on mock data and assert the grown
    tree is identical to the one built by sklearn.DecisionTreeRegressor.

    Parameters
    ----------
        random_state : int
        shape : list-like of int
        n_features : list-like of int
        nrules : int
        min_samples_leaf : int
        transpose_test : bool
        noise : float
    """
    print('Starting with settings:')
    pprint(dict(
        random_state=random_state,
        tree2_is_2d=tree2_is_2d,
        supervision=supervision,
        max_gen_depth=max_gen_depth,
    ) | params)

    with stopwatch():
        # XX, Y, x, y, gen_tree = make_interaction_regression(
        #     return_molten=True,
        #     return_tree=True,
        #     n_samples=params['n_samples'],
        #     n_features=params['n_features'],
        #     noise=params['noise'],
        #     random_state=random_state,
        #     max_depth=max_gen_depth,
        #     n_targets=params.get('n_targets'),
        #     max_target=1000,
        # )
        XX, Y, x, y = make_interaction_blobs(
            return_molten=True,
            n_samples=params['n_samples'],
            n_features=params['n_features'],
            random_state=random_state,
            noise=2.0,
            centers=10,
        )

    # NOTE on ExtraTrees:
    # Even with the same random_state, the way 2d splitter uses this random
    # state will be different (same random state for each axis), thus yielding
    # an ExtraTree2d different from sklearn's ExtraTree.

    with stopwatch(f'Fitting {tree1.__class__.__name__}...'):
        if supervision == 1.0:
            print('* Using supervised data for tree1.')
            tree1.fit(x, y)
        elif supervision == 0.0:
            print('* Using unsupervised data for tree1.')
            tree1.fit(x, x)
        else:
            print('* Using semisupervised data for tree1.')
            # Apply 'supervision' parameter weighting
            Xy = np.hstack((x.copy(), y.copy().reshape(-1, 1)))
            # Xy.shape[1] == n_features + n_outputs = n_features + 1
            Xy[:, -1] *= np.sqrt(supervision * Xy.shape[1])
            Xy[:, :-1] *= np.sqrt((1-supervision) *
                                  Xy.shape[1] / (Xy.shape[1]-1))
            tree1.fit(x, Xy)

    with stopwatch(f'Fitting {tree2.__class__.__name__}...'):
        if tree2_is_2d:
            print('* Using 2D data for tree2.')
            tree2.fit(XX, Y)
        else:
            print('* Using 1D data for tree2.')
            tree2.fit(x, y)
    
    if verbose:
        print(f'* Tree 1 ({tree1.__class__.__name__}) params:')
        pprint(tree1.get_params())
        print(f'* Tree 2 ({tree2.__class__.__name__}) params:')
        pprint(tree2.get_params())

    assert_equal_leaves(
        tree1.tree_,
        tree2.tree_,
        ignore=['value'] if supervision != 1.0 else None,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize('msl', [1, 5, 100])
@pytest.mark.parametrize('max_depth', [None, 5, 10])
@pytest.mark.parametrize(
    'criterion_mono, criterion_bi',
    [
        ('squared_error', 'squared_error_gso'),
        ('friedman_mse', 'friedman_gso'),
    ]
)
def test_simple_tree_1d2d(
    msl,
    random_state,
    max_depth,
    criterion_mono,
    criterion_bi,
    **params,
):
    params = DEF_PARAMS | params
    tree2 = BipartiteDecisionTreeRegressor(
        min_samples_leaf=msl,
        max_depth=max_depth,
        random_state=check_random_state(random_state),
        bipartite_adapter="gmosa",
        criterion=criterion_bi,
    )

    tree1 = DecisionTreeRegressor(
        min_samples_leaf=msl,
        max_depth=max_depth,
        criterion=criterion_mono,
        random_state=check_random_state(random_state),
    )

    return compare_trees(
        tree1=tree1,
        tree2=tree2,
        random_state=random_state,
        **params,
    )


@pytest.mark.parametrize(
    "pred_weight", [None, "uniform", "precomputed", lambda x: x**2])
class TestGMOSymmetry:
    """Test X symmetry constraints of GMO trees.
    """
    params = DEF_PARAMS
    XX, Y = make_interaction_regression(**params)
    XX_sim, Y_sim, = make_interaction_regression(
        **(params | dict(n_features=params['n_samples'])))

    pbct = BipartiteDecisionTreeRegressor(
        bipartite_adapter="gmo",
    )

    def test_square_array_error(self, pred_weight, **params):
        params = DEF_PARAMS | params
        pbct = clone(self.pbct).set_params(prediction_weights=pred_weight)
        with pytest.raises(
            ValueError,
            match=r"array must be 2-dimensional and square.",
        ):
            pbct.fit(self.XX, self.Y)

    def test_non_symmetric_array(self, pred_weight, **params):
        params = DEF_PARAMS | params
        pbct = clone(self.pbct).set_params(prediction_weights=pred_weight)
        with pytest.raises(
            ValueError,
            match=r"Array must be symmetric",
        ):
            pbct.fit(self.XX_sim, self.Y_sim)

    @pytest.mark.parametrize("splitter", ["random", "best"])
    def test_working(self, pred_weight, splitter, **params):
        params = DEF_PARAMS | params
        pbct = clone(self.pbct).set_params(
            prediction_weights=pred_weight,
            splitter=splitter,
        )
        XX_sim = [check_symmetric(Xi, raise_warning=False)
                  for Xi in self.XX_sim]
        pbct.fit(XX_sim, self.Y_sim)
        pbct.predict(np.hstack([XX_sim[0][:3], XX_sim[1][:3]]))


@pytest.mark.parametrize("splitter", ["random", "best"])
@pytest.mark.parametrize("adapter", ["gmosa"])
def test_identity_gso(splitter, adapter, gso_criterion, **params):
    params = DEF_PARAMS | params
    XX, Y, x, y = make_interaction_regression(return_molten=True, **params)
    tree = BipartiteDecisionTreeRegressor(
        min_samples_leaf=1,
        splitter=splitter,
        bipartite_adapter=adapter,
        criterion=gso_criterion,
    )
    assert_allclose(tree.fit(XX, Y).predict(XX).reshape(Y.shape), Y)


@pytest.mark.parametrize(
    "pred_weights", [
        None,
        *_bipartite_classes.PREDICTION_WEIGHTS_OPTIONS-{"raw"},
        lambda x: x ** 3,
    ]
)
@pytest.mark.parametrize("splitter", ["random", "best"])
@pytest.mark.parametrize(
    "criterion", _bipartite_classes.BIPARTITE_CRITERIA["gmo"]["regression"].keys()
)
def test_identity_gmo(pred_weights, splitter, criterion, **params):
    """Tests wether regression trees can grow util isolating every instance.
    """
    params = DEF_PARAMS | params
    params['n_features'] = params['n_samples'] #= (10, 10)

    XX, Y = make_interaction_regression(**params)
    tree = BipartiteDecisionTreeRegressor(
        min_samples_leaf=1,
        bipartite_adapter="gmo",
        prediction_weights=pred_weights,
        splitter=splitter,
        criterion=criterion,
    )
    XX = [check_symmetric(X, raise_warning=False) for X in XX]
    Y *= 1000

    tree.fit(XX, Y)

    wn_samples = tree.tree_.weighted_n_node_samples
    ch_left = tree.tree_.children_left
    ch_right = tree.tree_.children_right
    n_samples_per_leaf = wn_samples[ch_left == ch_right]

    assert_allclose(n_samples_per_leaf, 1.0)
    assert_allclose(tree.predict(XX).reshape(Y.shape), Y)


@pytest.mark.parametrize("splitter", ["random", "best"])
def test_random_state(splitter, n_samples, random_state):
    random_state = check_random_state(random_state)

    XX = [random_state.uniform(size=(n, n)) for n in n_samples]
    XX = [check_symmetric(X, raise_warning=False) for X in XX]
    Y = random_state.choice((0., 1.), size=n_samples)

    tree1 = BipartiteDecisionTreeRegressor(
        min_samples_leaf=1,
        bipartite_adapter="gmosa",
        splitter=splitter,
        criterion="squared_error",
        random_state=copy.deepcopy(random_state),
    ).fit(XX, Y)

    tree2 = BipartiteDecisionTreeRegressor(
        min_samples_leaf=1,
        bipartite_adapter="gmosa",
        splitter=splitter,
        criterion="squared_error",
        random_state=copy.deepcopy(random_state),
    ).fit(XX, Y)

    leaves1 = get_leaves(tree1.tree_)
    leaves2 = get_leaves(tree2.tree_)

    values1 = tree1.tree_.value[..., -1].reshape(-1)
    values2 = tree2.tree_.value[..., -1].reshape(-1)

    assert len(values1) > 10, "Not enough leaves"

    pred1 = tree1.predict(XX)
    pred2 = tree2.predict(XX)

    assert_equal_dicts(
        {
            **leaves1,
            'value': values1,
            'predictions': pred1,
        },
        {
            **leaves2,
            'value': values2,
            'predictions': pred2,
        },
    )


@pytest.mark.parametrize("splitter", ["random", "best"])
def test_gini_mse_identity(splitter, n_samples, random_state):
    random_state = check_random_state(random_state)

    XX = [random_state.uniform(size=(n, n)) for n in n_samples]
    XX = [check_symmetric(X, raise_warning=False) for X in XX]
    Y = random_state.choice((0., 1.), size=n_samples)

    tree_reg = BipartiteDecisionTreeRegressor(
        min_rows_leaf=10,
        min_cols_leaf=10,
        bipartite_adapter="gmosa",
        splitter=splitter,
        criterion="squared_error",
        random_state=copy.deepcopy(random_state),
    ).fit(XX, Y)

    tree_clf = BipartiteDecisionTreeClassifier(
        min_rows_leaf=10,
        min_cols_leaf=10,
        bipartite_adapter="gmosa",
        splitter=splitter,
        criterion="gini",
        random_state=copy.deepcopy(random_state),
    ).fit(XX, Y)

    leaves_reg = get_leaves(tree_reg.tree_)
    leaves_clf = get_leaves(tree_clf.tree_)

    leaves_reg['impurity'] *= 2

    values_reg = tree_reg.tree_.value[..., -1].reshape(-1)
    values_clf = tree_clf.tree_.value[..., -1].flatten()  # copy
    values_clf /= tree_clf.tree_.weighted_n_node_samples

    assert len(leaves_reg['impurity']) > 5, "Not enough leaves"

    pred_reg = tree_reg.predict(XX)
    pred_clf = tree_clf.predict_proba(XX)[..., -1]

    assert_equal_dicts(
        {
            **leaves_reg,
            'value': values_reg,
            'predictions': pred_reg,
        },
        {
            **leaves_clf,
            'value': values_clf,
            'predictions': pred_clf,
        },
    )


@pytest.mark.parametrize("min_samples_leaf", [1, 20, 100])
@pytest.mark.parametrize("splitter", ["random", "best"])
@pytest.mark.parametrize(
    "criterion", _bipartite_classes.BIPARTITE_CRITERIA["gmo"]["regression"].keys()
)
def test_leaf_mean_symmetry(min_samples_leaf, splitter, criterion):
    """Tests if Y_leaf.mean(0).mean() == Y_leaf.mean(1).mean()
    
    Y_leaf.mean(0) and Y_leaf.mean(1) are stored by regression GMO trees as
    output values. This function tests if these values are correct by checking
    the aforementioned equality.
    """
    params = DEF_PARAMS

    XX, Y, x, y = make_interaction_regression(return_molten=True, **params)
    tree = BipartiteDecisionTreeRegressor(
        min_samples_leaf=min_samples_leaf,
        bipartite_adapter="gmo",
        prediction_weights="raw",
        splitter=splitter,
        criterion=criterion,
    )
    tree.fit(XX, Y)
    pred = tree.predict(XX)
    row_pred, col_pred = np.hsplit(pred, [tree._n_rows_fit])
    mean_rows = np.nanmean(row_pred, axis=1)
    mean_cols = np.nanmean(col_pred, axis=1)

    assert not np.isnan(mean_rows).any()
    assert not np.isnan(mean_cols).any()
    assert_allclose(mean_rows, mean_cols)


def test_weight_normalization():
    rng = np.random.default_rng()
    w = rng.random((100, 20))
    zeroed_w = rng.choice(w.shape[0], w.shape[0]//10)
    w[zeroed_w] = 0.

    pred = rng.random((100, 20))
    not_nan = rng.choice((True, False), pred.shape)
    pred[~not_nan] = np.nan

    w = _normalize_weights(w, pred)

    assert_allclose(w.sum(axis=1), 1.)
    assert all(
        1. not in row or set(row.unique()) < {0., 1.}
        for row in w
    )
    assert np.all(w[np.isnan(pred)] == 0.)
    assert_allclose(
        w[zeroed_w],
        (not_nan / not_nan.sum(axis=1, keepdims=True))[zeroed_w]
    )


@pytest.mark.parametrize("mrl,mcl", [(1, 5), (2, 1), (11, 19)])
@pytest.mark.parametrize("criterion", ["squared_error"])
def test_leaf_shape_gmo(random_state, mrl, mcl, criterion, **params):
    params = DEF_PARAMS | params
    # XX, Y, x, y = gen_mock_data(melt=True, **params)
    rng = check_random_state(random_state)
    n_clusters = 10, 10
    Y_unique = np.arange(np.prod(n_clusters), dtype=float).reshape(n_clusters)
    Y = Y_unique.repeat(mrl, axis=0).repeat(mcl, axis=1)
    shape = mrl * n_clusters[0], mcl * n_clusters[1]

    n_features = 50, 50
    XX = [
        rng.random((s, f)).astype(np.float32)
        for s, f in zip(shape, n_features)
    ]
    XX[0][:, 0] = np.sort(XX[0][:, 0])
    XX[1][:, 0] = np.sort(XX[1][:, 0])

    # Shuffle
    id_rows = rng.choice(shape[0], size=shape[0], replace=False)
    id_cols = rng.choice(shape[1], size=shape[1], replace=False)
    XX[0] = XX[0][id_rows]
    XX[1] = XX[1][id_cols]
    Y = Y[np.ix_(id_rows, id_cols)]

    tree = BipartiteDecisionTreeRegressor(
        min_rows_leaf=mrl,
        min_cols_leaf=mcl,
        bipartite_adapter="gmo",
        prediction_weights="raw",
        criterion=criterion,
    )
    tree.fit(XX, Y)
    pred = tree.predict(XX)
    row_pred, col_pred = np.hsplit(pred, [tree._n_rows_fit])
    n_rows = (~np.isnan(row_pred)).sum(axis=1)
    n_cols = (~np.isnan(col_pred)).sum(axis=1)

    assert_allclose(n_rows, mrl)
    assert_allclose(n_cols, mcl)


@pytest.mark.parametrize("mrl,mcl", [(1, 1), (1, 5), (2, 1), (19, 11)])
def test_leaf_shape_gso(mrl, mcl, random_state, gso_criterion, **params):
    params = DEF_PARAMS | params
    # XX, Y, x, y = gen_mock_data(melt=True, **params)
    rng = check_random_state(random_state)
    n_clusters = 10, 10
    Y_unique = np.arange(np.prod(n_clusters), dtype=float).reshape(n_clusters)
    Y = Y_unique.repeat(mrl, axis=0).repeat(mcl, axis=1)
    y_values = np.sort(Y_unique.reshape(-1))

    shape = mrl * n_clusters[0], mcl * n_clusters[1]
    n_features = 50, 50
    XX = [
        rng.random((s, f)).astype(np.float32)
        for s, f in zip(shape, n_features)
    ]
    XX[0][:, 0] = np.sort(XX[0][:, 0])
    XX[1][:, 0] = np.sort(XX[1][:, 0])

    # Shuffle
    id_rows = rng.choice(shape[0], size=shape[0], replace=False)
    id_cols = rng.choice(shape[1], size=shape[1], replace=False)
    XX[0] = XX[0][id_rows]
    XX[1] = XX[1][id_cols]
    Y = Y[np.ix_(id_rows, id_cols)]

    # Control result
    X, y = row_cartesian_product(XX), Y.reshape(-1)
    tree1d = DecisionTreeRegressor()
    tree1d.fit(X, y)

    tree = BipartiteDecisionTreeRegressor(
        min_rows_leaf=mrl,
        min_cols_leaf=mcl,
        bipartite_adapter="gmosa",
        criterion=gso_criterion,
    )
    tree.fit(XX, Y)

    leaves1 = get_leaves(tree1d.tree_)
    leaf_values1d = np.sort(leaves1['value'].reshape(-1))
    assert_allclose(leaves1['n_node_samples'], mrl*mcl)
    assert_allclose(leaf_values1d, y_values)

    assert_equal_leaves(tree1d.tree_, tree.tree_)
