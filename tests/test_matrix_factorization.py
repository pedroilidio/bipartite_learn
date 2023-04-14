# TODO: rename
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.utils._testing import assert_allclose
from sklearn.utils.validation import check_symmetric
from imblearn.pipeline import make_pipeline

from bipartite_learn.matrix_factorization._nrlmf import NRLMF
from bipartite_learn.matrix_factorization._dnilmf import DNILMF
from bipartite_learn.preprocessing.multipartite import DTHybridSampler
from bipartite_learn.preprocessing.monopartite import TargetKernelLinearCombiner
from bipartite_learn.wrappers import (
    BipartiteLocalWrapper, MultipartiteSamplerWrapper,
)
from .utils.test_utils import gen_mock_data, DEF_PARAMS, parse_args


def test_nrlmf(**params):
    params = DEF_PARAMS | params
    params['noise'] = 0
    params["shape"] = params["nattrs"]  # X must be kernel matrices

    XX, Y, = gen_mock_data(**params, melt=False)

    nrlmf = NRLMF(verbose=True)
    XXt, Yt = nrlmf.fit_resample(XX, Y)
    nrlmf.predict([XX[0][:5], XX[1][:3]])
    Y_positive = Y.astype(bool)

    assert Yt[Y_positive].mean() > Yt[~Y_positive].mean()

    return XXt


def test_nrlmf_fit_predict(**params):
    params = DEF_PARAMS | params
    params['noise'] = 0
    params["shape"] = params["nattrs"]  # X must be kernel matrices

    XX, Y, = gen_mock_data(**params, melt=False)
    XX = [check_symmetric(Xi, raise_warning=False) for Xi in XX]
    np.fill_diagonal(XX[0], 1.)
    np.fill_diagonal(XX[1], 1.)

    nrlmf = NRLMF(verbose=True, random_state=0)

    pred1 = nrlmf.fit(XX, Y).predict(XX)
    U1, V1 = nrlmf.U, nrlmf.V

    pred2 = nrlmf.fit_predict(XX, Y)
    U2, V2 = nrlmf.U, nrlmf.V

    assert_allclose(U1, U2)
    assert_allclose(V1, V2)
    assert_allclose(pred1, pred2)


def test_dnilmf_fit_predict(**params):
    params = DEF_PARAMS | params
    params['noise'] = 0
    params["shape"] = params["nattrs"]  # X must be kernel matrices

    XX, Y, = gen_mock_data(**params, melt=False)
    XX = [check_symmetric(Xi, raise_warning=False) for Xi in XX]
    np.fill_diagonal(XX[0], 1.)
    np.fill_diagonal(XX[1], 1.)

    dnilmf = DNILMF(verbose=True, random_state=0)
    pred1 = dnilmf.fit(XX, Y).predict(XX)
    U1, V1 = dnilmf.U, dnilmf.V

    pred2 = dnilmf.fit_predict(XX, Y)
    U2, V2 = dnilmf.U, dnilmf.V

    assert_allclose(U1, U2)
    assert_allclose(V1, V2)
    assert_allclose(pred1, pred2)


def test_dnilmf(**params):
    params = DEF_PARAMS | params
    params['noise'] = 0
    params["shape"] = params["nattrs"]  # X must be kernel matrices

    XX, Y, = gen_mock_data(**params, melt=False)

    dnilmf = DNILMF(verbose=True)
    XXt, Yt = dnilmf.fit_resample(XX, Y)
    dnilmf.predict([XX[0][:5], XX[1][:3]])
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
    gip_transformer = MultipartiteSamplerWrapper(TargetKernelLinearCombiner())

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
    test_nrlmf_fit_predict(**params)
    test_dnilmf(**params)
    test_dnilmf_fit_predict(**params)
    test_dthybrid(**params)
    test_blmnii(**params)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))