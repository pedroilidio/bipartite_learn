from .utils.test_utils import gen_mock_data


def test_make_interaction_function(**params):
    params.update(dict(
        shape=(100, 100),
        nattrs=(1, 1),
        nrules=15,
        noise=0,
        seed=0,
        transpose_test=False,
    ))
    X, y = gen_mock_data(**params)

    assert not any(y.all(axis=a).any() for a in range(y.ndim))