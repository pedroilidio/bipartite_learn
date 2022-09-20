from make_examples import make_binary_interaction_func
from test_utils import gen_mock_data, DEF_PARAMS, parse_args


def test_make_interaction_function(**params):
    params.update(dict(
        shape = (100, 100),
        nattrs = (1, 1),
        nrules = 15,
        noise = 0,
    ))
    X, y = gen_mock_data(**params)

    assert not any(y.all(axis=a).any() for a in range(y.ndim))


def main(**params):
    test_make_interaction_function(**params)


if __name__ == "__main__":
    args = parse_args()
    main(**(DEF_PARAMS | vars(args)))