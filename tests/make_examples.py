from argparse import ArgumentParser
from itertools import product, pairwise
from pathlib import Path
from pprint import pprint
import numpy as np
from collections import defaultdict
from hypertrees.melter import row_cartesian_product

DIR_HERE = Path(__file__).resolve().parent


def _check_random_state(random_state):
    if not isinstance(random_state, np.random.Generator):
        return np.random.default_rng(random_state)
    return random_state


def parse_args(args=None):
    argparser = ArgumentParser()
    argparser.add_argument('--shape', nargs='+', type=int, default=[1000, 800])
    argparser.add_argument('--nattrs', nargs='+', type=int, default=[200, 300])
    argparser.add_argument('--nrules', type=int, default=10)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('-o', '--outdir', type=Path,
                           default=Path(DIR_HERE.parent/'examples/input'))
    return argparser.parse_args(args)


def make_interaction_data(
         shape, nattrs, func=None, nrules=5, quiet=False, noise=0.,
         random_state=None, return_intervals=False,
 ):
    random_state = _check_random_state(random_state)

    if func is None:
        func, intervals = make_binary_interaction_func(
            np.sum(nattrs), nrules, random_state=random_state)

    # shape contains the number of instances in each axis database, i.e.
    # its number of rows. nattrs contains their numbers of columns, i.e.
    # how many attributes per axis.
    print("Generating X...")
    XX = [random_state.random((ni, nj), dtype=np.float32)
          for ni, nj in zip(shape, nattrs)]
    X = row_cartesian_product(XX)
    
    print("Generating y...")
    y = np.apply_along_axis(func, -1, X)

    if noise:
        y = y.astype(float)
        y += noise * random_state.random(y.size)

    if func is not None and return_intervals:
        return XX, y.reshape(shape), intervals
    return XX, y.reshape(shape)


def make_intervals(
    nattrs, n_rules, random_state=None,
):
    rng = _check_random_state(random_state)
    boundaries = defaultdict(lambda: [0, 1])

    for _ in range(n_rules):
        boundaries[rng.integers(nattrs)].append(rng.random())
    
    intervals = {}
    for attr, bounds in boundaries.items():
        intervals[attr] = list(pairwise(sorted(bounds)))

    print("Generated decision boundaries:")
    pprint(dict(boundaries))
    
    return intervals

    
def make_binary_interaction_func(
    nattrs, n_rules, random_state=None,
):
    rng = _check_random_state(random_state)
    intervals = make_intervals(nattrs, n_rules, random_state)

    attrs = list(intervals.keys())
    interv = list(intervals.values())
    invert = rng.choice(2)
    
    def interaction_func(x):
        indices = (range(len(i)) for i in interv)

        for ii, region in zip(product(*indices), product(*interv)):
            for attr, interval in zip(attrs, region):
                if not (interval[0] <= x[attr] < interval[1]):
                    break
            else:
                return (sum(ii) + invert) % 2

        raise ValueError("x values must be between 0 and 1")
    
    return interaction_func, intervals


def make_dense_interaction_func(
    nattrs, n_boundaries, random_state=None,
):
    rng = _check_random_state(random_state)
    intervals = make_intervals(nattrs, n_rules, random_state)
    features = list(intervals.keys())
    seed = rng.random()
    
    def interaction_func(x):
        inner_rng = np.random.default_rng(seed)

        for region in product(*intervals.values()):
            return_value = inner_rng.random()

            for attr, interval in zip(features, region):
                if not (interval[0] <= x[attr] < interval[1]):
                    break
            else:
                return return_value

        raise ValueError("x values must be between 0 and 1")
    
    return interaction_func, intervals
