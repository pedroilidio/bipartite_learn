# TODO: rename
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from imblearn.pipeline import make_pipeline

from hypertrees.matrix_factorization._nrlmf import NRLMF
from hypertrees.matrix_factorization._dnilmf import DNILMF
from hypertrees.preprocessing.multipartite import DTHybridSampler
from hypertrees.preprocessing.monopartite import TargetKernelLinearCombiner
from hypertrees.wrappers import (
    BipartiteLocalWrapper, MultipartiteTransformerWrapper,
)
from test_utils import gen_mock_data, DEF_PARAMS, parse_args


def test_nrlmf(**params):
    params = DEF_PARAMS | params
    params['noise'] = 0
    params["shape"] = params["nattrs"]  # X must be kernel matrices

    XX, Y, = gen_mock_data(**params, melt=False)

    nrlmf = NRLMF(verbose=True)
    XXt, Yt = nrlmf.fit_resample(XX, Y)
    Y_positive = Y.astype(bool)

    assert Yt[Y_positive].mean() > Yt[~Y_positive].mean()

    return XXt


def test_dnilmf(**params):
    params = DEF_PARAMS | params
    params['noise'] = 0
    params["shape"] = params["nattrs"]  # X must be kernel matrices

    XX, Y, = gen_mock_data(**params, melt=False)

    dnilmf = DNILMF(verbose=True)
    XXt, Yt = dnilmf.fit_resample(XX, Y)
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

    dthybrid = DTHybridSampler()
    XXt, Yt = dthybrid.fit_resample(XX, Y)
    Y_positive = Y.astype(bool)

    assert Yt[Y_positive].mean() > Yt[~Y_positive].mean()


def test_blmnii(**params):
    params = DEF_PARAMS | params
    params["noise"] = 0
    params["shape"] = params["nattrs"]  # X must be kernel matrices

    # Gaussian interaction profile
    gip_transformer = MultipartiteTransformerWrapper(TargetKernelLinearCombiner())

    XX, Y, = gen_mock_data(**params, melt=False)

    blmnii = make_pipeline(
        gip_transformer,
        BipartiteLocalWrapper(
            primary_estimator=KNeighborsRegressor(),
            secondary_estimator=Ridge(),
        ),
    )

    Yt = blmnii.fit(XX, Y).predict(XX).reshape(Y.shape)
    Y_positive = Y.astype(bool)

    assert Yt[Y_positive].mean() > Yt[~Y_positive].mean()


def main(**params):
    test_nrlmf(**params)
    test_dnilmf(**params)
    test_dthybrid(**params)
    test_blmnii(**params)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))