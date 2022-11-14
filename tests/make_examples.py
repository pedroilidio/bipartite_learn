from argparse import ArgumentParser
from itertools import product, pairwise
from pathlib import Path
from pprint import pprint
import numpy as np
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils import check_random_state
from collections import defaultdict
from hypertrees.melter import row_cartesian_product

DIR_HERE = Path(__file__).resolve().parent


def oversample_from_bipartite_tree2(
    tree, n_row_features, n_samples=(1, 1), random_state=None,
):
    rng = check_random_state(random_state)

    is_internal = tree.children_left != tree.children_right
    features = tree.feature[is_internal]
    thresholds = tree.threshold[is_internal]

    idx = features.argsort()
    feature_values, group_ids = np.unique(features[idx], return_index=True)

    groups = dict(zip(
        feature_values,
        np.split(thresholds[idx], group_ids[1:])
    ))

    new_values = [[], []]

    for feature in range(tree.n_features):
        thresh = list(groups.get(feature, []))  # feature threshold values
        axis = feature >= n_row_features

        for _ in range(n_samples[axis]):
            new_values[axis] += list(rng.uniform(
                low=[0] + thresh[:-1],
                high=thresh[0:-1] + [1],
            ))

    return new_values


def oversample_from_tree3(tree, n_samples=(1, 1), random_state=None):
    rng = check_random_state(random_state)

    is_internal = tree.children_left != tree.children_right
    features = tree.feature[is_internal]
    thresholds = tree.threshold[is_internal]

    idx = features.argsort()
    groups = np.split(
        thresholds[idx],
        np.unique(features[idx], return_index=True)[1][1:],  # skip 0
    )
    groups = [
        np.insert(a, obj=(0, a.size), values=(0., 1.))
        for a in groups
    ]

    out = []
    for _ in range(n):
        feature_values = [rng.uniform(low=a[1:], high=a[:-1]) for a in groups]

    return groups


def _oversample_from_bipartite_tree(
    tree, pos, n, axis, n_row_features, random_state, low, high, result
):
    child_left, child_right = tree.children_left[pos], tree.children_right[pos]

    if child_left == child_right:  # leaf reached
        if axis == pos >= n_row_features:
            for _ in range(n):  # append n samples from the leaf
                result.append(random_state.uniform(low=low, high=high))
            return
        return

    low_left = low.copy()
    low_right = low.copy()
    high_left = high.copy()
    high_right = high.copy()

    if axis == pos >= n_row_features:
        feature = tree.feature[pos]
        threshold = tree.threshold[pos]
        low_right[feature] = threshold
        high_left[feature] = threshold

        _oversample_from_bipartite_tree(
            tree, child_left, n, axis, n_row_features, random_state,
            low_left, high_left, result)
        return

    _oversample_from_bipartite_tree(
        tree, child_right, n, axis, n_row_features, random_state, low_right,
        high_right, result)


def oversample_from_bipartite_tree(
    tree,
    n_row_features,
    random_state=None,
):
    random_state = check_random_state(random_state)
    n_col_features = tree.n_features - n_row_features

    low_rows = np.zeros(n_row_features, dtype=float)
    high_rows = np.ones(n_row_features, dtype=float)

    low_cols = np.zeros(n_col_features, dtype=float)
    high_cols = np.ones(n_col_features, dtype=float)

    res_rows, res_cols = [], []

    _oversample_from_bipartite_tree(
        tree, 0, n, n_row_features, random_state, low_rows,
        high_rows, res_rows, axis=0)
    _oversample_from_bipartite_tree(
        tree, 0, n, n_row_features, random_state, low_cols,
        high_cols, res_cols, axis=1)
    
    return [res_rows, res_cols]



def _oversample_from_tree(tree, pos, n, random_state, low, high, result):
    child_left, child_right = tree.children_left[pos], tree.children_right[pos]

    if child_left == child_right:  # leaf reached
        for _ in range(n):  # append n samples from the leaf
            result.append(random_state.uniform(low=low, high=high))
        return

    low_left = low.copy()
    low_right = low.copy()
    high_left = high.copy()
    high_right = high.copy()

    feature = tree.feature[pos]
    threshold = tree.threshold[pos]
    low_right[feature] = threshold
    high_left[feature] = threshold

    _oversample_from_tree(tree, child_left, n, random_state, low_left,
                          high_left, result)
    _oversample_from_tree(tree, child_right, n, random_state, low_right,
                          high_right, result)


def oversample_from_tree(tree, n=1, random_state=None):
    """Returns n new samples for each leaf.

    Parameters
    ----------
    tree : sklearn.tree.Tree instance
        Tree object representing a fitted tree estimator
    n : int, optional
        The amount of new samples to draw for each lea, by default 1
    random_state : int, np.RandomState, None, optional
        Seed or random state object, by default None

    Returns
    -------
    list
        list of arrays with n new samples per leaf
    """
    random_state = check_random_state(random_state)
    low = np.zeros(tree.n_features, dtype=float)
    high = np.ones(tree.n_features, dtype=float)
    res = []
    _oversample_from_tree(tree, 0, n, random_state, low, high, res)
    return np.asarray(res)


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
    random_state = check_random_state(random_state)

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
    random_state = check_random_state(random_state)
    boundaries = defaultdict(lambda: [0, 1])

    for _ in range(n_rules):
        boundaries[random_state.integers(nattrs)].append(random_state.random())
    
    intervals = {}
    for attr, bounds in boundaries.items():
        intervals[attr] = list(pairwise(sorted(bounds)))

    print("Generated decision boundaries:")
    pprint(dict(boundaries))
    
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
    nattrs, n_boundaries, random_state=None,
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


def make_interaction_regression(
    n_samples=100,
    n_features=50,
    n_targets=None,
    min_target=0.0,
    max_target=1.0,
    return_tree=False,
    stratified=False,
    shuffle=True,
    max_depth=None,
    random_state=None,
):
    if stratified and n_targets is None:
        raise ValueError("If stratified=True, provide n_targets")

    if isinstance(n_samples, int):
        n_samples = (n_samples, n_samples)
    if isinstance(n_features, int):
        n_features = (n_features, n_features)
    if isinstance(n_targets, int):
        n_targets = (n_targets, n_targets)

    random_state = check_random_state(random_state)
    n_targets = n_targets or n_samples

    # X = [random_state.random((s, f)) for s, f in zip(shape, n_features)]
    X = [random_state.random((s, f)) for s, f in zip(n_targets, n_features)]
    xx = row_cartesian_product(X)
    y = (max_target - min_target) * random_state.random(np.prod(n_targets))

    tree = ExtraTreeRegressor(
        min_samples_leaf=1,
        max_features=1,
        max_depth=max_depth,
    )
    tree.fit(xx, y)

    if not stratified:
        X = [random_state.random((s, f)) for s, f in zip(n_samples, n_features)]
        Y = tree.predict(row_cartesian_product(X)).reshape(n_samples)
        if return_tree:
            return X, Y, tree.tree_
        return X, Y

    if n_targets != n_samples:
        if n_samples[0] % n_targets[0] != 0:
            raise ValueError(
                f"{n_samples[0]=} must be divisible by {n_targets[0]=}")
        if n_samples[1] % n_targets[1] != 0:
            raise ValueError(
                f"{n_samples[1]=} must be divisible by {n_targets[1]=}")

        leaf_shape = (
            n_samples[0]//n_targets[0],
            n_samples[1]//n_targets[1],
        )
        n = max(leaf_shape)-1
        new_samples = oversample_from_tree(
            tree.tree_, n=n, random_state=random_state)

        new_X = [[X[0]], [X[1]]]
        for i in range(0, len(new_samples), n):
            leaf_samples = new_samples[i:i+n]
            new_X_rows, new_X_cols = np.split(
                leaf_samples, n_features[:-1], axis=1)

            # Take just the necessary
            new_X_rows = new_X_rows[:leaf_shape[0]-1]
            new_X_cols = new_X_cols[:leaf_shape[1]-1]

            new_X[0].append(new_X_rows)
            new_X[1].append(new_X_cols)

        new_X[0] = np.vstack(new_X[0])
        new_X[1] = np.vstack(new_X[1])
        breakpoint()

        # TODO: no need to predict again
        X = new_X

    Y = tree.predict(row_cartesian_product(X)).reshape(n_samples)

    if shuffle:
        id_rows = random_state.choice(
            n_samples[0], size=n_samples[0], replace=False)
        id_cols = random_state.choice(
            n_samples[1], size=n_samples[1], replace=False)

        X = [X[0][id_rows], X[1][id_cols]]
        Y = Y[np.ix_(id_rows, id_cols)]

    if return_tree:
        return X, Y, tree.tree_
    return X, Y
