from bipartite_learn import BipartiteDecisionTreeRegressor
from .utils.make_examples import make_interaction_blobs
from .utils.test_utils import stopwatch
from .test_bipartite_trees import assert_equal_leaves
from time import perf_counter


def main():
    result_gmo = []
    result_gso = []

    for n_samples, n_features in (
        ((5, 5), (50, 50)),
        ((10, 9), (50, 100)),
        ((100, 90), (50, 40)),
        ((100, 90), (50, 40)),
        ((10, 10), (100, 200)),
        ((10, 10), (500, 400)),
        ((10, 10), (5000, 4000)),
        ((10, 10), (50000, 40000)),
    ):
        X, Y = make_interaction_blobs(
            random_state=0,
            n_samples=n_samples,
            n_features=n_features,
            noise=2.0,
            centers=10,
        )

        print('Training GMO tree')
        t0 = perf_counter()
        tree_gmo = BipartiteDecisionTreeRegressor(
            criterion='squared_error_gso',
            bipartite_adapter='gmosa',
        ).fit(X, Y)
        result_gmo.append(perf_counter() - t0)

        print('Training GSO tree')
        t0 = perf_counter()
        tree_gso = BipartiteDecisionTreeRegressor(
            criterion='squared_error',
            bipartite_adapter='gso',
        ).fit(X, Y)
        result_gso.append(perf_counter() - t0)

        assert_equal_leaves(tree_gmo.tree_, tree_gso.tree_)

        print('GMO:', result_gmo)
        print('GSO:', result_gso)


if __name__ == '__main__':
    main()