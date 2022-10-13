import logging
import numpy as np
from itertools import product
from pprint import pprint
from typing import Callable

import pytest
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
import sklearn.tree

from hypertrees.tree._nd_classes import (
    DecisionTreeRegressor2D,
    BiclusteringTreeRegressor,
)
from hypertrees.melter import row_cartesian_product
from hypertrees.melter import row_cartesian_product

from make_examples import make_interaction_data
from test_utils import stopwatch, parse_args, gen_mock_data, melt_2d_data

# from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64

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
)


def print_eval_model(tree, XX, Y):
    x_gen = (np.hstack(x).reshape(1, -1) for x in product(*XX))
    pred = np.fromiter((tree.predict(x) for x in x_gen), dtype=float, like=Y)
    print('Final MSE:', np.mean((pred.reshape(-1) - Y.reshape(-1))**2))
    print('R^2:', np.corrcoef(pred.reshape(-1), Y.reshape(-1))[0, 1] ** 2)
    fake_preds = Y.reshape(-1).copy()
    np.random.shuffle(fake_preds)
    print('Random baseline MSE:', np.mean((fake_preds - Y.reshape(-1))**2))
    print('Random baseline R^2:',
          np.corrcoef(fake_preds, Y.reshape(-1))[0, 1] ** 2)


def print_n_samples_in_leaves(tree):
    wn_samples = tree.tree_.weighted_n_node_samples
    ch_left = tree.tree_.children_left
    ch_right = tree.tree_.children_right
    n_samples_per_leaf = wn_samples[ch_left == ch_right]

    print('==> Class name:', tree.__class__.__name__)
    print('Estimator params:')
    pprint(tree.get_params())
    print('n_nodes:', tree.tree_.node_count)
    print('n_leaves:', n_samples_per_leaf.shape[0])
    print('weighted_n_node_samples:', n_samples_per_leaf)
    return n_samples_per_leaf


# TODO: parameter description.
def compare_trees(
    tree1=DecisionTreeRegressor,
    tree2=DecisionTreeRegressor2D,
    tree2_is_2d=True,
    tree1_is_unsupervised=False,
    **params,
):
    """Test hypertreesDecisionTreeRegressor2D

    Fit hypertreesDecisionTreeRegressor2D on mock data and assert the grown
    tree is identical to the one built by sklearn.DecisionTreeRegressor.

    Parameters
    ----------
        seed : int
        shape : list-like of int
        nattrs : list-like of int
        nrules : int
        min_samples_leaf : int
        transpose_test : bool
        noise : float
        inspect : bool
        plot : bool
    """
    print('Starting with settings:')
    pprint(params)

    with stopwatch():
        XX, Y, x, y = gen_mock_data(melt=True, **params)

    # ######### Instantiate trees
    if isinstance(tree2, Callable):
        tree2 = tree2(
            min_samples_leaf=params['min_samples_leaf'],
            random_state=params['seed'],
        )

    if isinstance(tree1, Callable):
        tree1 = tree1(
            min_samples_leaf=params['min_samples_leaf'],
            random_state=params['seed'],
        )
    # NOTE on ExtraTrees:
    # Even with the same random_state, the way 2d splitter uses this random
    # state will be different (same random state for each axis), thus yielding
    # an ExtraTree2d different from sklearn's ExtraTree.

    with stopwatch(f'Fitting {tree1.__class__.__name__}...'):
        if tree1_is_unsupervised:
            print('Using unsupervised data for tree1.')
            tree1.fit(x, x)
        else:
            tree1.fit(x, y)

    with stopwatch(f'Fitting {tree2.__class__.__name__}...'):
        if tree2_is_2d:
            print('Using 2D data for tree2.')
            tree2.fit(XX, Y)
        else:
            print('Using 1D data for tree2.')
            tree2.fit(x, y)

    tree1_n_samples_in_leaves = print_n_samples_in_leaves(tree1)
    tree2_n_samples_in_leaves = print_n_samples_in_leaves(tree2)

    if params['inspect']:
        breakpoint()

    # =========================================================================
    # Start of comparison tests
    # =========================================================================

    leaves_comparison = \
        tree1_n_samples_in_leaves == tree2_n_samples_in_leaves

    # comparison_test = np.all(leaves_comparison)  # See issue #1
    comparison_test = \
        set(tree1_n_samples_in_leaves) == set(tree2_n_samples_in_leaves)

    stree1 = sklearn.tree.export_text(tree1)
    stree2 = sklearn.tree.export_text(tree2)

    if params['plot']:
        with open('tree1.txt', 'w') as f:
            f.write(stree1)
        with open('tree2.txt', 'w') as f:
            f.write(stree2)

    assert (tree1_n_samples_in_leaves.shape[0] ==
            tree2_n_samples_in_leaves.shape[0]), \
        "Number of leaves differ."
    assert (tree1_n_samples_in_leaves.sum() ==
            tree2_n_samples_in_leaves.sum()), \
        "Total weighted_n_samples of leaves differ."

    assert comparison_test, (
        "Some leaves differ in the number of samples."
        f"\n\tdiff positions: {np.nonzero(~leaves_comparison)[0]}"
        f"\n\tdiff: {tree1_n_samples_in_leaves[~leaves_comparison]}"
        f"!= {tree2_n_samples_in_leaves[~leaves_comparison]}")


def test_simple_tree_1d2d(
    tree1=DecisionTreeRegressor,
    tree2=DecisionTreeRegressor2D,
    **params,
):
    params = DEF_PARAMS | params
    return compare_trees(
        tree1=DecisionTreeRegressor,
        tree2=DecisionTreeRegressor2D,
        **params,
    )


def test_pbct_regressor(**params):
    params = DEF_PARAMS | params

    print('Starting with settings:')
    pprint(params)

    with stopwatch():
        XX, Y = gen_mock_data(**params)

    for pred_weight in (
        "uniform",
        np.random.rand(sum(Y.shape)),
    ):
        pbct = BiclusteringTreeRegressor(prediction_weights=pred_weight)
        pbct = clone(pbct)
        print("*** Passed cloning test.")
        pbct.fit(XX, Y)
        print_n_samples_in_leaves(pbct)
        print(pbct.predict(np.hstack([XX[0][:3], XX[1][:3]])))

    with stopwatch():
        XX_sim, Y_sim = gen_mock_data(**(params|dict(nattrs=params['shape'])))

    for pred_weight in ("x", 3., lambda A: A**2):
        pbct = BiclusteringTreeRegressor(prediction_weights=pred_weight)
        with pytest.raises(ValueError, match=r"square \(pairwise\)"):
            pbct.fit(XX, Y)
        pbct = clone(pbct)
        print("*** Passed cloning test.")
        pbct.fit(XX_sim, Y_sim)
        print_n_samples_in_leaves(pbct)
        print(pbct.predict(np.hstack([XX_sim[0][:3], XX_sim[1][:3]])))


def main(**params):
    test_simple_tree_1d2d(**params)
    test_pbct_regressor(**params)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
