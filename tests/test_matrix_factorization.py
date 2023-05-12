# TODO: rename
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.utils._testing import assert_allclose
from sklearn.utils.validation import check_symmetric
from sklearn.metrics.pairwise import rbf_kernel
from imblearn.pipeline import make_pipeline

from bipartite_learn.matrix_factorization._nrlmf import NRLMF
from bipartite_learn.matrix_factorization._dnilmf import DNILMF
from bipartite_learn.preprocessing.multipartite import DTHybridSampler
from bipartite_learn.preprocessing.monopartite import TargetKernelLinearCombiner
from bipartite_learn.wrappers import (
    LocalMultiOutputWrapper, MultipartiteSamplerWrapper,
)
from .utils.test_utils import gen_mock_data, DEF_PARAMS, parse_args


@pytest.fixture
def data():
    params = DEF_PARAMS.copy()
    params['noise'] = 0
    XX, Y, *_ = gen_mock_data(**params, melt=False)
    XX = [rbf_kernel(Xi, gamma=1) for Xi in XX]
    Y = (Y > Y.mean()).astype('float64')
    return XX, Y


def test_nrlmf(data):
    XX, Y = data
    nrlmf = NRLMF(verbose=True)
    XXt, Yt = nrlmf.fit_resample(XX, Y)
    nrlmf.predict([XX[0][:5], XX[1][:3]])
    Y_positive = Y.astype(bool)

    assert Yt[Y_positive].mean() > Yt[~Y_positive].mean()

    return XXt


def test_nrlmf_fit_predict(data):
    XX, Y = data

    nrlmf = NRLMF(verbose=True, random_state=0)

    pred1 = nrlmf.fit(XX, Y).predict(XX)
    U1, V1 = nrlmf.U, nrlmf.V

    pred2 = nrlmf.fit_predict(XX, Y)
    U2, V2 = nrlmf.U, nrlmf.V

    assert_allclose(U1, U2)
    assert_allclose(V1, V2)
    assert_allclose(pred1, pred2)


def test_dnilmf_fit_predict(data):
    XX, Y = data

    dnilmf = DNILMF(verbose=True, random_state=0)
    pred1 = dnilmf.fit(XX, Y).predict(XX)
    U1, V1 = dnilmf.U, dnilmf.V

    pred2 = dnilmf.fit_predict(XX, Y)
    U2, V2 = dnilmf.U, dnilmf.V

    assert_allclose(U1, U2)
    assert_allclose(V1, V2)
    assert_allclose(pred1, pred2)


def test_dnilmf(data):
    XX, Y = data

    dnilmf = DNILMF(verbose=True)
    XXt, Yt = dnilmf.fit_resample(XX, Y)
    dnilmf.predict([XX[0][:5], XX[1][:3]])
    Y_positive = Y.astype(bool)

    assert Yt[Y_positive].mean() > Yt[~Y_positive].mean()

    return XXt


def test_dthybrid(data):
    XX, Y = data
    dthybrid = DTHybridSampler()
    XXt, Yt = dthybrid.fit_resample(XX, Y)
    Y_positive = Y.astype(bool)

    assert Yt[Y_positive].mean() > Yt[~Y_positive].mean()


def test_blmnii(data):
    XX, Y = data

    # Gaussian interaction profile
    gip_transformer = MultipartiteSamplerWrapper(TargetKernelLinearCombiner())

    blmnii = make_pipeline(
        gip_transformer,
        LocalMultiOutputWrapper(
            primary_rows_estimator=KNeighborsRegressor(),
            primary_cols_estimator=KNeighborsRegressor(),
            secondary_rows_estimator=Ridge(),
            secondary_cols_estimator=Ridge(),
        ),
    )

    Yt = blmnii.fit(XX, Y).predict(XX).reshape(Y.shape)
    Y_positive = Y.astype(bool)

    assert Yt[Y_positive].mean() > Yt[~Y_positive].mean()
