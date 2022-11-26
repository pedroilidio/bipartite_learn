from numbers import Integral, Real
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import validate_params, Interval
from ..melter import row_cartesian_product


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
