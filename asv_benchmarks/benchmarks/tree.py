from itertools import product
from hypertrees.tree._nd_criterion import GlobalMSE
from hypertrees.tree._nd_classes import BipartiteDecisionTreeRegressor
from hypertrees.datasets import make_interaction_regression


class TreeBenchmark:
    param_names = ["tree_params", "n_samples", "n_features"]
    repeat = 30
    params = (
        [
            dict(
                bipartite_adapter="local_multioutput",
                criterion=GlobalMSE,
                prediction_weights="leaf_uniform",
            ),
            dict(
                bipartite_adapter="global_single_output",
                criterion="squared_error",
                prediction_weights=None,
            ),
        ],
        [(30, 20), (50, 70), (200, 100), (200, 300)],
        [(10, 9)],
    )

    def setup(self, tree_params, n_samples, n_features):
        self.XX, self.Y = make_interaction_regression(
            n_samples=n_samples,
            n_features=n_features,
        )
        self.tree = BipartiteDecisionTreeRegressor(**tree_params)
        self.tree.fit(self.XX, self.Y)

    def time_fit(self, tree_params, n_samples, n_features):
        self.tree.fit(self.XX, self.Y)

    def time_predict(self, tree_params, n_samples, n_features):
        self.tree.predict(self.XX)


if __name__ == '__main__':

    for args in product(*TreeBenchmark.params):
        bench = TreeBenchmark()
        bench.setup(*args)
        bench.time_fit(*args)
        bench.time_predict(*args)