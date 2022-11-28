from argparse import ArgumentParser
from collections import defaultdict
from itertools import product, pairwise
from numbers import Integral, Real
from pathlib import Path
from pprint import pprint
import numpy as np
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import validate_params, Interval
from hypertrees.melter import row_cartesian_product

DIR_HERE = Path(__file__).resolve().parent


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
    random_state=None, return_intervals=False, verbose=False,
 ):
    random_state = check_random_state(random_state)

    if func is None:
        func, intervals = make_binary_interaction_func(
            np.sum(nattrs), nrules, random_state=random_state)

    # shape contains the number of instances in each axis database, i.e.
    # its number of rows. nattrs contains their numbers of columns, i.e.
    # how many attributes per axis.
    if verbose:
        print("Generating X...")
    XX = [random_state.rand(ni, nj) for ni, nj in zip(shape, nattrs)]
    X = row_cartesian_product(XX)

    if verbose:
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
    random_state = check_random_state(random_state)
    boundaries = defaultdict(lambda: [0, 1])

    for _ in range(n_rules):
        boundaries[random_state.randint(nattrs)].append(random_state.random())

    intervals = {}
    for attr, bounds in boundaries.items():
        intervals[attr] = list(pairwise(sorted(bounds)))

    # print("Generated decision boundaries:")
    # pprint(dict(boundaries))

    return intervals


def make_binary_interaction_func(
    nattrs, n_rules, random_state=None,
):
    random_state = check_random_state(random_state)
    intervals = make_intervals(nattrs, n_rules, random_state)

    attrs = list(intervals.keys())
    interv = list(intervals.values())
    invert = random_state.choice(2)

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
    nattrs, n_boundaries, random_state=None, n_rules=None,
):
    random_state = check_random_state(random_state)
    intervals = make_intervals(nattrs, n_rules, random_state)
    features = list(intervals.keys())
    seed = random_state.random()
    # Y, *_ = make_checkerboard(
    #     shape=shape,
    #     n_clusters=n_clusters,
    #     random_state=params['seed'],
    #     shuffle=False,
    # )

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


@validate_params(dict(
    n_samples=[list, tuple, Interval(Integral, 1, None, closed="left")],
    n_features=[list, tuple, Interval(Integral, 1, None, closed="left")],
    n_targets=[list, tuple, Interval(Integral, 1, None, closed="left"), None],
    min_target=[list, tuple, Real],
    max_target=[list, tuple, Real],
    max_depth=[Interval(Integral, 1, None, closed="left"), None],
    noise=[Interval(Real, 0.0, None, closed="left")],
    return_molten=["boolean"],
    return_tree=["boolean"],
    random_state=["random_state"],
))
def make_interaction_regression(
    n_samples=100,
    n_features=50,
    n_targets=None,
    min_target=0.0,
    max_target=1.0,
    noise=0.0,
    return_molten=False,
    return_tree=False,
    random_state=None,
    max_depth=None,
):
    if isinstance(n_samples, int):
        n_samples = (n_samples, n_samples)
    if isinstance(n_features, int):
        n_features = (n_features, n_features)
    if isinstance(n_targets, int):
        n_targets = (n_targets, n_targets)

    random_state = check_random_state(random_state)
    n_targets = n_targets or n_samples

    X = [random_state.random((s, f)) for s, f in zip(n_targets, n_features)]
    y = (
        random_state.random(n_targets)
        * (max_target - min_target)
        + min_target
    )

    tree = ExtraTreeRegressor(
        min_samples_leaf=1,
        max_features=1,
        max_depth=max_depth,
    ).fit(
        row_cartesian_product(X),
        y.reshape(-1),
    )

    # Make new data
    X = [random_state.random((s, f)) for s, f in zip(n_samples, n_features)]
    X_molten = row_cartesian_product(X)
    Y_molten = tree.predict(X_molten)
    if noise > 0.0:
        Y_molten += random_state.normal(scale=noise, size=Y_molten.size)
    Y = Y_molten.reshape(n_samples)

    ret = [X, Y]

    if return_molten:
        ret += [X_molten, Y_molten]
    if return_tree:
        ret.append(tree.tree_)

    return tuple(ret)
