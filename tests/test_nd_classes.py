from argparse import ArgumentParser
from itertools import product
from pprint import pprint
# from patched_modules._criterion import MSE
# from patched_modules._splitter import BestSplitter
from sklearn.tree._criterion import MSE
from sklearn.tree._splitter import BestSplitter
from sklearn.tree import DecisionTreeRegressor
import sklearn.tree

from hypertree.tree._nd_splitter import Splitter2D, make_2d_splitter
from hypertree.tree._nd_criterion import MSE_Wrapper2D
from hypertree.tree._nd_classes import DecisionTreeRegressor2D
from hypertree.melter import row_cartesian_product

import numpy as np
#from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64

from pathlib import Path
import sys
from time import time

from make_examples import gen_imatrix

import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# Default test params
DEF_CONFIG = dict(
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
    save_trees=False,
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
    print('Random baseline R^2:', np.corrcoef(fake_preds, Y.reshape(-1))[0, 1] ** 2)


def print_n_samples_in_leaves(tree):
    wn_samples = tree.tree_.weighted_n_node_samples
    ch_left = tree.tree_.children_left
    ch_right = tree.tree_.children_right
    n_samples_per_leaf = wn_samples[ch_left == ch_right]

    print('n_nodes', tree.tree_.node_count)
    print('n_leaves', n_samples_per_leaf.shape[0])
    print('weighted_n_node_samples:', n_samples_per_leaf)
    return n_samples_per_leaf


def parse_args(**DEF_CONFIG):
    argparser = ArgumentParser(fromfile_prefix_chars='@')
    argparser.add_argument('--seed', type=int)
    argparser.add_argument('--shape', nargs='+', type=int)
    argparser.add_argument('--nattrs', nargs='+', type=int)
    argparser.add_argument('--nrules', type=int)
    argparser.add_argument('--min_samples_leaf', '--msl', type=int)

    argparser.add_argument('--transpose_test', action='store_true')
    argparser.add_argument('--noise', type=float)
    argparser.add_argument('--inspect', action='store_true')
    argparser.add_argument('--plot', action='store_true')
    argparser.add_argument('--save_trees', action='store_true')
    argparser.set_defaults(**DEF_CONFIG)

    return argparser.parse_args()


# TODO: parameter description.
def main(**CONFIG):
    """Test hypertree.DecisionTreeRegressor2D

    Fit hypertree.DecisionTreeRegressor2D on mock data and assert the grown tree
    is identical to the one built by sklearn.DecisionTreeRegressor.

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
        save_trees : bool
    """
    ## Generate mock data
    print('Starting with settings:')
    pprint(CONFIG)

    np.random.seed(CONFIG['seed'])
    # rng = np.random.default_rng(CONFIG['seed)  # should use.

    t0 = time()
    XX, Y = gen_imatrix(CONFIG['shape'], CONFIG['nattrs'], nrules=CONFIG['nrules'])

    if CONFIG['transpose_test']:
        print('Test transposing axis.')
        Y = np.copy(Y.T.astype(DOUBLE_t), order='C')
        XX = [np.ascontiguousarray(X, DTYPE_t) for X in XX[::-1]]
        CONFIG['nattrs'] = CONFIG['nattrs'][::-1]
        CONFIG['shape'] = CONFIG['shape'][::-1]
    else:
        Y = np.ascontiguousarray(Y, DOUBLE_t)
        XX = [np.ascontiguousarray(X, DTYPE_t) for X in XX]

    if CONFIG['noise']:
        Y += np.random.rand(*CONFIG['shape']) * CONFIG['noise'] # add some noise

    print('Data generation time:', time()-t0)
    print('Data density (mean):', Y.mean())
    print('Data variance:', Y.var())
    print('=' * 50)

    ########## Instantiate trees
    tree2d = DecisionTreeRegressor2D(
        min_samples_leaf=CONFIG['min_samples_leaf'],
    )
    tree1d = DecisionTreeRegressor(
        min_samples_leaf=CONFIG['min_samples_leaf'],
    )

    t0 = time()
    print('Fitting sklearn.DecisionTreeRegressor...')
    tree2d.fit(XX, Y)
    print(f'Done in {time()-t0} s.')

    t0 = time()
    print('Fitting 1D tree...')
    tree1d.fit(row_cartesian_product(XX), Y.reshape(-1))
    print(f'Done in {time()-t0} s.')

    tree2d_n_samples_in_leaves = print_n_samples_in_leaves(tree2d)
    tree1d_n_samples_in_leaves = print_n_samples_in_leaves(tree1d)

    if not CONFIG['inspect']:
        assert np.all(
            tree1d_n_samples_in_leaves == tree2d_n_samples_in_leaves
        )
        assert (
            sklearn.tree.export_text(tree2d) == sklearn.tree.export_text(tree1d)
        )

    # print('Evaluating 2D tree...')
    # print_eval_model(tree2d, XX, Y)
    # print_n_samples_in_leaves(tree)
    # print('Evaluating 1D tree...')
    # print_eval_model(tree1d, XX, Y)
    # print_n_samples_in_leaves(tree1d)

    with open('tree1d.txt', 'w') as f:
        f.write(sklearn.tree.export_text(tree1d))
    with open('tree2d.txt', 'w') as f:
        f.write(sklearn.tree.export_text(tree2d))

    if CONFIG['plot']:
        import matplotlib.pyplot as plt
        sklearn.tree.plot_tree(tree2d)
        plt.show()

    if CONFIG['inspect']:
        breakpoint()


def test_main():
    main(**DEF_CONFIG)


if __name__ == "__main__":
    args = parse_args(**DEF_CONFIG)
    main(**vars(args))
