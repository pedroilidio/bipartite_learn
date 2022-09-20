import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from imblearn.pipeline import make_pipeline

from hypertrees.matrix_factorization._nrlmf import NRLMF
from hypertrees.preprocessing.bipartite_samplers import (
    DTHybrid, BipartiteRBFSampler,
)
from hypertrees.wrappers import BipartiteLocalWrapper
from test_utils import gen_mock_data, DEF_PARAMS, parse_args


def test_nrlmf(**params):
    params = DEF_PARAMS | params
    params['noise'] = 0
    params["shape"] = params["nattrs"]  # X must be kernel matrices

    XX, Y, = gen_mock_data(**params, melt=False)

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

    shape, nattrs = params["shape"], params["nattrs"]
    XX = [
        rng.random((shape[0], nattrs[0])),
        rng.random((shape[1], nattrs[1])),
    ]
    Y = rng.choice((0., 1.), size=shape)

    dthybrid = DTHybrid()
    XXt, Yt = dthybrid.fit_resample(XX, Y)
    Y_positive = Y.astype(bool)

    assert Yt[Y_positive].mean() > Yt[~Y_positive].mean()


def test_blmnii(**params):
    params = DEF_PARAMS | params
    params["noise"] = 0
    params["shape"] = params["nattrs"]  # X must be kernel matrices

    XX, Y, = gen_mock_data(**params, melt=False)

    blmnii = make_pipeline(
        BipartiteRBFSampler(),
        BipartiteLocalWrapper(
            estimator_rows=KNeighborsRegressor(),
            estimator_cols=KNeighborsRegressor(),
            secondary_estimator_rows=Ridge(),
            secondary_estimator_cols=Ridge(),
        ),
    )

    Yt = blmnii.fit(XX, Y).predict(XX).reshape(Y.shape)
    Y_positive = Y.astype(bool)

    assert Yt[Y_positive].mean() > Yt[~Y_positive].mean()


def main(**params):
    # test_nrlmf(**params)
    # test_dthybrid(**params)
    test_blmnii(**params)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))