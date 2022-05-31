from make_examples import make_interaction_data
from test_utils import stopwatch, parse_args, gen_mock_data

import logging
from itertools import product
from pprint import pprint
from sklearn.tree import DecisionTreeRegressor
import sklearn.tree

from hypertrees.tree._nd_classes import DecisionTreeRegressor2D
from hypertrees.melter import row_cartesian_product

import numpy as np
# from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# Default test params
DEF_PARAMS = dict(
    # seed=439,
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
# FIXME: Some leafs do not coincide whith the parameters below:
# --seed 23 --noise .1 --nrules 20 --shape 500 600 --nattrs 10 9 --msl 100


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

    print('n_nodes', tree.tree_.node_count)
    print('n_leaves', n_samples_per_leaf.shape[0])
    print('weighted_n_node_samples:', n_samples_per_leaf)
    return n_samples_per_leaf


# TODO: parameter description.
def compare_trees(
    tree1=DecisionTreeRegressor,
    tree2=DecisionTreeRegressor2D,
    tree2_is_2d=True,
    tree2_is_ss=False,
    **PARAMS,
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
    pprint(PARAMS)

    with stopwatch():
        XX, Y, strfunc = gen_mock_data(**PARAMS)

    # ######### Instantiate trees
    if isinstance(tree2, type):
        tree2 = tree2(
            min_samples_leaf=PARAMS['min_samples_leaf'],
            # splitter='random',
            random_state=PARAMS['seed'],
        )

    if isinstance(tree1, type):
        tree1 = tree1(
            min_samples_leaf=PARAMS['min_samples_leaf'],
            random_state=PARAMS['seed'],
        )
    # NOTE on ExtraTrees:
    # Even with the same random_state, the way 2d splitter uses this random
    # state will be different (same random state for each axis), thus yielding
    # an ExtraTree2d different from sklearn's ExtraTree.

    with stopwatch(f'Fitting {tree2.__class__.__name__}...'):
        if tree2_is_2d:
            print('Using 2D data.')
            tree2.fit(XX, Y)
        else:
            print('Using 1D data.')
            X, y = row_cartesian_product(XX), Y.reshape(-1, 1)
            if tree2_is_ss:
                tree2.fit(X, np.hstack((X, y)))
            else:
                tree2.fit(X, y)

    with stopwatch(f'Fitting {tree1.__class__.__name__}...'):
        tree1.fit(row_cartesian_product(XX), Y.reshape(-1))

    tree1_n_samples_in_leaves = print_n_samples_in_leaves(tree1)
    tree2_n_samples_in_leaves = print_n_samples_in_leaves(tree2)

    if PARAMS['plot']:
        stree1 = sklearn.tree.export_text(tree1)
        stree2 = sklearn.tree.export_text(tree2)

        with open('tree1.txt', 'w') as f:
            f.write(stree1)
        with open('tree2.txt', 'w') as f:
            f.write(stree2)

        # import matplotlib.pyplot as plt
        # sklearn.tree.plot_tree(tree2)
        # plt.show()


    if not PARAMS['inspect']:
        assert tree1_n_samples_in_leaves.shape[0] == \
               tree2_n_samples_in_leaves.shape[0]

        leaves_comparison = \
            tree1_n_samples_in_leaves == tree2_n_samples_in_leaves
        comparison_test = np.all(leaves_comparison)
        if not comparison_test:
            print("diff positions:", np.nonzero(~leaves_comparison)[0])
            print("diff: ", tree1_n_samples_in_leaves[~leaves_comparison],
                  '!=', tree2_n_samples_in_leaves[~leaves_comparison])

        assert comparison_test
        assert (
            sklearn.tree.export_text(tree2) == sklearn.tree.export_text(tree1)
        )

    if PARAMS['inspect']:
        breakpoint()


def main(
    tree1=DecisionTreeRegressor,
    tree2=DecisionTreeRegressor2D,
    **PARAMS,
):
    return compare_trees(
        tree1=DecisionTreeRegressor,
        tree2=DecisionTreeRegressor2D,
        **PARAMS,
    )


def test_main():
    main(**DEF_PARAMS)


if __name__ == "__main__":
    args = parse_args(**DEF_PARAMS)
    main(**vars(args))
