from statistics import mean
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.base import clone
from hypertrees.wrappers import PU_WrapperND
from test_utils import gen_mock_data, parse_args

# Default test params
DEF_PARAMS = dict(
    # seed=439,
    seed=7,
    shape=(50, 60),
    nattrs=(10, 9),
    nrules=2,
    min_samples_leaf=1,
    transpose_test=False,
    noise=0.1,
    inspect=False,
    plot=False,
    max_seed_attempts=100,
)


def test_pu_wrapper_nd_no_neg_subsamples(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS | {'noise': 0}
    XX, Y, X, y, _ = gen_mock_data(melt=True, **PARAMS)
    Y = (Y == 1).astype(int)
    y = (y == 1).astype(int)

    tree = DecisionTreeRegressor(
        min_samples_leaf=PARAMS['min_samples_leaf'],
        random_state=PARAMS['seed'],
    )
    tree_nd = PU_WrapperND(clone(tree), subsample_negatives=False)

    tree_nd.fit(XX, Y)
    tree.fit(X, y)

    assert export_text(tree_nd.estimator) == export_text(tree)


def test_pu_wrapper_nd_neg_subsamples(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS | {'noise': 0}

    for _ in range(PARAMS['max_seed_attempts']):
        XX, Y, X, y, _ = gen_mock_data(melt=True, **PARAMS)
        if y.mean() < .5:
            break
        else:
            print(f"Seed {PARAMS['seed']} generated too much positive values,"
                  " we need less than half of the total. Trying next seed.")
            PARAMS['seed'] += 1
    else:
        raise ValueError("Could not find a valid seed. Max attempts reached "
                         f"({PARAMS['max_seed_attempts']}). You could choose "
                         "a different start seed (--seed) or increase max. "
                         "attempts (--max_seed_attempts).")

    tree = DecisionTreeRegressor(
        min_samples_leaf=PARAMS['min_samples_leaf'],
        random_state=PARAMS['seed'],
    )
    tree_nd = PU_WrapperND(clone(tree), subsample_negatives=True)
    Xs, ys = tree_nd._melt_Xy(XX, Y)

    print(f"Keeping {ys.shape[0]} out of {y.shape[0]}. "
          f"Relative new size: {ys.shape[0]/y.shape[0]}\n")

    assert ys.mean() == .5
    assert ys.shape[0]/y.shape[0] == 2*y.mean()
    assert ys.shape[0] < y.shape[0]
    assert Xs.shape[0] == ys.shape[0]

    tree_nd.fit(XX, Y)
    tree.fit(Xs, ys)

    # print(export_text(tree_nd.estimator))
    # print(export_text(tree))

    assert export_text(tree_nd.estimator) == export_text(tree)


def main(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    test_pu_wrapper_nd_no_neg_subsamples(**PARAMS)
    test_pu_wrapper_nd_neg_subsamples(**PARAMS)


if __name__ == '__main__':
    params = vars(parse_args(**DEF_PARAMS))
    main(**params)
    print('Done.')