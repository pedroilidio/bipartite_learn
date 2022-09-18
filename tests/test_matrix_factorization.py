import numpy as np
from hypertrees.matrix_factorization._nrlmf import NRLMF, DTHybrid
from test_utils import gen_mock_data, DEF_PARAMS, parse_args


def test_nrlmf(**params):
    params = DEF_PARAMS | params
    params['noise'] = 0
    params["shape"] = params["nattrs"]  # X must be kernel matrices

    while True:
        XX, Y, _ = gen_mock_data(**params, melt=False)
        if Y.var():
            break
        params['seed'] += 1
        print('New seed:', params['seed'])

    nrlmf = NRLMF()
    XXt, Yt = nrlmf.fit_resample(XX, Y)
    Y_positive = Y.astype(bool)

    assert Yt[Y_positive].mean() > Yt[~Y_positive].mean()

    return XXt


def test_dthybrid(**params):
    params = DEF_PARAMS | params
    params["noise"] = 0
    params["shape"] = params["nattrs"]  # X must be kernel matrices
    rng = np.random.default_rng(params["seed"])

    # while True:
    #     XX, Y, _ = gen_mock_data(**params, melt=False)
    #     if Y.var():
    #         break
    #     params["seed"] += 1
    #     print("New seed:", params["seed"])

    shape, nattrs = params["shape"], params["nattrs"]
    XX = [
        rng.random((shape[0], nattrs[0])),
        rng.random((shape[1], nattrs[1])),
    ]
    Y = rng.choice((0., 1.), size=shape)

    dthybrid = DTHybrid()
    XXt, Yt = dthybrid.fit_resample(XX, Y)
    Y_positive = Y.astype(bool)

    print(Y, Yt)
    assert Yt[Y_positive].mean() > Yt[~Y_positive].mean()


def main(**params):
    # test_nrlmf(**params)
    test_dthybrid(**params)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))