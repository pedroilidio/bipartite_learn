import joblib
import pytest
from pathlib import Path
import numpy as np
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


def test_gso_wrapper_bipartite_subsampling(random_state, min_samples_leaf):
    Y_ = (Y == 1).astype(int)
    y_ = (y == 1).astype(int)

    tree = DecisionTreeRegressor(
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree_bipartite = GlobalSingleOutputWrapper(clone(tree))

    tree_bipartite.fit(XX, Y_)
    tree.fit(X, y_)

    assert export_text(tree_bipartite.estimator_) == export_text(tree)


def _test_pickling(obj, tmp_path):
    joblib.dump(obj, tmp_path/'test.pickle')
    obj_loaded = joblib.load(tmp_path/'test.pickle')
    obj.get_params()
    obj_loaded.get_params()


def test_wrapped_tree_pickling(tmp_path):
    tree = DecisionTreeRegressor(
        min_samples_leaf=30,
        random_state=0,
    )
    tree_bipartite = GlobalSingleOutputWrapper(clone(tree))

    _test_pickling(tree, tmp_path)
    _test_pickling(tree_bipartite, tmp_path)


def test_wrapped_forest_pickling(tmp_path):
    forest = RandomForestRegressor(
        min_samples_leaf=30,
        random_state=0,
    )
    forest_bipartite = GlobalSingleOutputWrapper(
        clone(forest)
    )

    _test_pickling(forest, tmp_path)
    _test_pickling(forest_bipartite, tmp_path)


def test_score():
    forest = RandomForestRegressor(
        min_samples_leaf=30,
        random_state=0,
    )
    forest_bipartite = GlobalSingleOutputWrapper(
        clone(forest),
    )
    forest_bipartite.fit(XX, Y)

    score = _score(
        forest_bipartite, X, y,
        check_scoring(forest_bipartite, "roc_auc"),
        error_score="raise",
    )
    print("Score:", score)


def test_gso():
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.under_sampling import RandomUnderSampler
    from bipartite_learn.datasets import NuclearReceptorsLoader
    X, y = NuclearReceptorsLoader().load()  # X is a list of two matrices

    bipartite_clf = GlobalSingleOutputWrapper(
        estimator=RandomForestClassifier(),
        under_sampler=RandomUnderSampler(),
    )
    bipartite_clf.fit(X, y)
    assert bipartite_clf.score(X, y) > 0.5


def test_gmo():
    from bipartite_learn.datasets import NuclearReceptorsLoader
    from bipartite_learn.wrappers import LocalMultiOutputWrapper
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.multioutput import MultiOutputClassifier
    
    X, y = NuclearReceptorsLoader().load()  # X is a list of two matrices
    bipartite_clf = LocalMultiOutputWrapper(
        primary_rows_estimator=MultiOutputClassifier(SVC()),
        primary_cols_estimator=MultiOutputClassifier(SVC()),
        secondary_rows_estimator=KNeighborsClassifier(),
        secondary_cols_estimator=KNeighborsClassifier(),
        combine_predictions_func=np.max,
    )
    bipartite_clf.fit(X, y)
    assert bipartite_clf.score(X, y) > 0.5