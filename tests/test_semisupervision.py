from make_examples import make_interaction_data
from test_nd_classes import compare_trees, parse_args

from argparse import ArgumentParser
from itertools import product
from pprint import pprint

from sklearn.tree import DecisionTreeRegressor

import sys
sys.path.append(__file__+'/..')
from hypertrees.melter import row_cartesian_product
from hypertrees.tree._semisupervised_criterion import (
    SSMSE,
)

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


def main(
    **PARAMS,
):
    tree2 = DecisionTreeRegressor(
        criterion=SSMSE(
            n_features=PARAMS['nattrs'][0],
            n_outputs=1,
            n_samples=PARAMS['shape'][0],
            supervision=1,
        ),
        min_samples_leaf=PARAMS['min_samples_leaf'],
        random_state=PARAMS['seed'],
    )
    return compare_trees(
        tree1=DecisionTreeRegressor,
        tree2=tree2,
        tree2_is_2d=False,
        tree2_is_ss=True,
        **PARAMS,
    )


def test_main():
    main(**DEF_PARAMS)


if __name__ == "__main__":
    args = parse_args(**DEF_PARAMS)
    main(**vars(args))
