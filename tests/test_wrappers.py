import joblib
import pytest
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.model_selection._validation import _score
from sklearn.metrics import check_scoring
from bipartite_learn.wrappers import GlobalSingleOutputWrapper
from bipartite_learn.melter import melt_multipartite_dataset
from .utils.make_examples import make_interaction_regression
from .utils.test_utils import stopwatch

with stopwatch('Making dataset'):
    XX, Y, X, y = make_interaction_regression(
        return_molten=True,
        n_samples=(50, 60),
        n_features=(10, 9),
        random_state=0,
        max_target=1.0,
    )
    # Make it binary  # TODO: specialized data generation function
    y_mean = Y.mean()
    Y = (Y > y_mean).astype(int)
    y = (y > y_mean).astype(int)
    # Ensure negative majority
    if Y.mean() > 0.5:
        Y, y = 1-Y, 1-y


@pytest.fixture(params=range(10))
def random_state(request):
    return request.param


@pytest.fixture(params=[1, 0.1])
def min_samples_leaf(request):
    return request.param


def test_pu_wrapper_bipartite_no_neg_subsamples(random_state, min_samples_leaf):
    Y_ = (Y == 1).astype(int)
    y_ = (y == 1).astype(int)

    tree = DecisionTreeRegressor(
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree_bipartite = GlobalSingleOutputWrapper(clone(tree), subsample_negatives=False)

    tree_bipartite.fit(XX, Y_)
    tree.fit(X, y_)

    assert export_text(tree_bipartite.estimator) == export_text(tree)


def test_pu_wrapper_bipartite_neg_subsamples(random_state, min_samples_leaf):
    tree = DecisionTreeRegressor(
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree_bipartite = GlobalSingleOutputWrapper(
        estimator=clone(tree),
        subsample_negatives=True,
        random_state=random_state,
    )
    Xs, ys = melt_multipartite_dataset(
        XX, Y,
        subsample_negatives=True,
        random_state=random_state,
    )

    print(f"Keeping {ys.shape[0]} out of {y.shape[0]}. "
          f"Relative new size: {ys.shape[0]/y.shape[0]}\n")

    assert ys.mean() == .5
    assert ys.shape[0]/y.shape[0] == 2*y.mean()
    assert ys.shape[0] < y.shape[0]
    assert Xs.shape[0] == ys.shape[0]

    tree_bipartite.fit(XX, Y)
    tree.fit(Xs, ys)

    # print(export_text(tree_bipartite.estimator))
    # print(export_text(tree))

    assert export_text(tree_bipartite.estimator) == export_text(tree)


def _test_pickling(obj):
    joblib.dump(obj, 'test.pickle')
    obj_loaded = joblib.load('test.pickle')
    Path('test.pickle').unlink()

    obj.get_params()
    obj_loaded.get_params()


def test_wrapped_tree_pickling():
    tree = DecisionTreeRegressor(
        min_samples_leaf=30,
        random_state=0,
    )
    tree_bipartite = GlobalSingleOutputWrapper(clone(tree), subsample_negatives=True)

    _test_pickling(tree)
    _test_pickling(tree_bipartite)


def test_wrapped_forest_pickling():
    forest = RandomForestRegressor(
        min_samples_leaf=30,
        random_state=0,
    )
    forest_bipartite = GlobalSingleOutputWrapper(
        clone(forest), subsample_negatives=True)

    _test_pickling(forest)
    _test_pickling(forest_bipartite)


def test_score():
    forest = RandomForestRegressor(
        min_samples_leaf=30,
        random_state=0,
    )
    forest_bipartite = GlobalSingleOutputWrapper(
        clone(forest),
        subsample_negatives=True,
        random_state=0,
    )
    forest_bipartite.fit(XX, Y)

    score = _score(
        forest_bipartite, X, y,
        check_scoring(forest_bipartite, "roc_auc"),
        error_score="raise",
    )
    print("Score:", score)
