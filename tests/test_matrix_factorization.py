import pytest
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.utils._testing import assert_allclose
from sklearn.utils.validation import check_symmetric
from sklearn.metrics.pairwise import rbf_kernel
from imblearn.pipeline import make_pipeline

from bipartite_learn.matrix_factorization._nrlmf import (
    NRLMFSampler, NRLMFClassifier, NRLMFTransformer,
)
from bipartite_learn.matrix_factorization._dnilmf import (
    DNILMFSampler, DNILMFClassifier, DNILMFTransformer,
)
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


@pytest.mark.parametrize(
    'estimator', [
        NRLMFSampler(verbose=True, random_state=0),
        DNILMFSampler(verbose=True, random_state=0),
    ],
    ids=['nrlmf', 'dnilmf'],
)
def test_mf_sampler(data, estimator):
    XX, Y = data
    XXt, Yt = estimator.fit_resample(XX, Y)
    Y_positive = Y.astype(bool)
    assert Yt[Y_positive].mean() > Yt[~Y_positive].mean()


@pytest.mark.parametrize(
    'estimator', [
        NRLMFClassifier(verbose=True, random_state=0),
        DNILMFClassifier(verbose=True, random_state=0),
    ],
    ids=['nrlmf', 'dnilmf'],
)
def test_mf_classifier(data, estimator):
    XX, Y = data

    pred1 = estimator.fit(XX, Y).predict_proba(XX)
    U1, V1 = estimator.U, estimator.V

    pred2 = estimator.fit_predict_proba(XX, Y)
    U2, V2 = estimator.U, estimator.V

    assert_allclose(U1, U2)
    assert_allclose(V1, V2)
    assert_allclose(pred1, pred2)

    estimator.predict(XX).shape == (Y.size,)
    estimator.fit_predict(XX, Y).shape == (Y.size,)



@pytest.mark.parametrize(
    'estimator', [
        NRLMFTransformer(verbose=True, random_state=0),
        DNILMFTransformer(verbose=True, random_state=0),
    ],
    ids=['nrlmf', 'dnilmf'],
)
def test_mf_transformer(data, estimator):
    XX, Y = data
    estimator.fit(XX, Y)
    XXt = estimator.transform(XX)
    XXt2 = estimator.fit_transform(XX, Y)

    assert np.allclose(estimator.U, XXt[0])
    assert np.allclose(estimator.V, XXt[1])
    assert np.allclose(estimator.U, XXt2[0])
    assert np.allclose(estimator.V, XXt2[1])


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
