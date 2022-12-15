# TODO: test with different axis supervisions. sscrit2d.impurity_improvement()
#       may fail.
from pprint import pprint
from time import time
from pathlib import Path
from typing import Callable
import warnings
import pytest
from numbers import Real, Integral, Number

import numpy as np
from pytest import raises

from sklearn.tree._criterion import MSE, MAE, FriedmanMSE, Poisson
from sklearn.tree._splitter import BestSplitter, RandomSplitter
from sklearn.utils.validation import check_random_state
from sklearn.utils._testing import assert_allclose
from sklearn.utils._param_validation import Interval, validate_params

from hypertrees.tree._splitter_factory import (
    make_2d_splitter,
    make_2dss_splitter,
    make_semisupervised_criterion,
)
from hypertrees.tree._nd_criterion import (
    PBCTCriterionWrapper,
    MSE_Wrapper2D,
    FriedmanAdapter,
    RegressionCriterionWrapper2D,
)
from hypertrees.tree._axis_criterion import AxisMSE
from hypertrees.tree._semisupervised_criterion import (
    SSCompositeCriterion,
    # SingleFeatureSSCompositeCriterion,
)
from hypertrees.tree._semisupervised_splitter import BestSplitterSFSS
from hypertrees.tree._experimental_criterion import UD3, UD35
from make_examples import make_interaction_blobs, make_interaction_regression

from splitter_test import test_splitter, test_splitter_nd
from test_utils import (
    parse_args, stopwatch, gen_mock_data, melt_2d_data, assert_equal_dicts,
)
from test_criteria import manual_split_eval_mse

#from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64


# Default test params
DEF_PARAMS = dict(
    seed=0,
    # shape=(10, 10),
    shape=(50, 50),
    # nattrs=(10, 9),
    nattrs=(10, 10),
    nrules=1,
    min_samples_leaf=1,
    # min_samples_leaf=100,
    transpose_test=False,
    noise=.5,
    inspect=False,
    plot=False,
    start=0,
    end=-1,
    verbose=False,
)


# since we need to convert X from float32 to float64, the actual precision for
# unsupervised data is 1e-4, so that smaller multiples will yield differencees.
@pytest.fixture(
    params=[0.0, 0.001, 0.1, 0.2328, 0.569, 0.782, 0.995, 1.0]
)
def supervision(request):
    return request.param


@pytest.fixture(
    params=range(10)
)
def random_state(request):
    return request.param


def compare_splitters_1d2d_ideal(
    splitter1,
    splitter2,
    tol=0,
    **params,
):
    params = DEF_PARAMS | dict(noise=0) | params

    if params['noise']:
        warnings.warn(f"noise={params['noise']}. Setting it to zero"
                      " since noise=0 is what defines an ideal split.")
        params['noise'] = 0

    result1, result2 = compare_splitters_1d2d(
        splitter1, splitter2, tol, **params)

    assert result1['improvement'] != 0
    assert result1['impurity_left'] == 0
    assert result1['impurity_right'] == 0

    assert result2['improvement'] != 0
    assert result2['impurity_left'] == 0
    assert result2['impurity_right'] == 0


# TODO: Refactoring needed.
def compare_splitters_1d2d(
    splitter1,
    splitter2,
    semisupervised_1d=False,
    unsupervised_1d=False,
    single_feature_ss_1d=False,
    multioutput_1d=False,
    only_1d=False,
    manual_impurity=True,
    rtol=1e-7,
    atol=0.0,
    **params,
):
    params = DEF_PARAMS | params
    random_state = check_random_state(params['seed'])

    if params['nrules'] != 1:
        warnings.warn(f"Setting params['nrules'] = 1 (was {params['nrules']})")
        params['nrules'] = 1

    # XX, Y, x, y, _ = gen_mock_data(**params, melt=True)
    XX, Y, intervals = gen_mock_data(return_intervals=True, **params)
    n_samples_0 = params['shape'][0]

    # Artificially set XX[0][:, 0] to be the best (most separable)
    # unsupervised attribute.
    XX[0][:, 0] = np.hstack([
        random_state.normal(loc=.2, scale=.1, size=n_samples_0//2),
        random_state.normal(loc=.8, scale=.1, size=n_samples_0-n_samples_0//2),
    ])

    if Y.var() == 0:
        raise RuntimeError(f"Bad seed ({params['seed']}), y is homogeneous."
                           " Try another one or reduce nrules.")

    if multioutput_1d:  # Test GMO splitter
        # The split will be across rows (ax = 0) or columns (ax = 1)
        ax = next(iter(intervals.keys())) >= XX[0].shape[1]
        x, y = XX[int(ax)], np.ascontiguousarray(Y.T) if ax else Y
        start = 0  # TODO: param
        end = y.shape[0]  # TODO: param
        start1d, end1d = start, end
        start2d = [0, start] if ax else [start, 0]
        end2d = [Y.shape[0], end] if ax else [end, Y.shape[1]]
    else:
        x, y = melt_2d_data(XX, Y)
        start = 0  # TODO: param
        end = Y.shape[0]  # TODO: param
        start1d = start*Y.shape[1]
        end1d = end*Y.shape[1]
        start2d = [start, 0]
        end2d = [end, Y.shape[1]]

    if single_feature_ss_1d:
        x = x[:, 0].reshape(-1, 1)
    elif semisupervised_1d:
        y = np.hstack([x, y]).astype(DOUBLE_t)
    elif unsupervised_1d:
        y = np.ascontiguousarray(x, dtype=DOUBLE_t)

    print(x.shape, y.shape)

    if (not isinstance(start, int)) or (not isinstance(end, int)):
        raise TypeError(f"2D start/end not possible. start ({repr(start)}) and end "
                        f"({repr(end)}) must be integers.")

    if isinstance(splitter1, Callable):
        splitter1 = splitter1(
            criterion=MSE(n_outputs=y.shape[1], n_samples=x.shape[0]),
            max_features=x.shape[1],
            min_samples_leaf=1,
            min_weight_leaf=0,
            random_state=check_random_state(params['seed']),
        )

    if isinstance(splitter2, Callable):
        if only_1d:
            splitter2 = splitter2(
                criterion=MSE(n_outputs=y.shape[1], n_samples=x.shape[0]),
                max_features=x.shape[1],
                min_samples_leaf=y.shape[1],
                min_weight_leaf=0,
                random_state=check_random_state(params['seed']),
            )
        else:
            splitter2 = make_2d_splitter(
                splitters=splitter2,
                criteria=MSE,
                max_features=[X.shape[1] for X in XX],
                n_samples=Y.shape,
                n_outputs=1,
                random_state=check_random_state(params['seed']),
            )

    # Run test
    with stopwatch(f'Testing 1D splitter ({splitter1.__class__.__name__})...'):
        result1 = test_splitter(
            splitter1, x, y,
            start=start1d,
            end=end1d,
            verbose=params['verbose'],
        )
        print('Best split found:')
        pprint(result1)

    assert result1['improvement'] >= 0, \
        'Negative reference improvement, input seems wrong.'

    if manual_impurity:
        x_ = x[start1d:end1d]
        y_ = y[start1d:end1d]
        pos = result1['pos'] - start1d

        if semisupervised_1d:
            # NOTE: only single-output
            x_, y_ = x_.copy(), y_.copy()
            # Apply 'supervision' parameter weighting
            sup = splitter1.criterion.supervision
            # y_.shape[1] == n_features + n_outputs = n_features + 1
            y_[:, -1] *= np.sqrt(sup * y_.shape[1])
            y_[:, :-1] *= np.sqrt((1-sup) * y_.shape[1] / (y_.shape[1]-1))

        sorted_indices = x_[:, result1['feature']].argsort()
        manual_impurity_left = y_[sorted_indices][:pos].var(0).mean()
        manual_impurity_right = y_[sorted_indices][pos:].var(0).mean()

        print(f'* manual_impurity_left={manual_impurity_left}')
        print(f'* manual_impurity_right={manual_impurity_right}')

        assert_allclose(
            result1['impurity_left'],
            manual_impurity_left,
            err_msg='Wrong reference impurity left.',
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            result1['impurity_right'],
            manual_impurity_right,
            err_msg='Wrong reference impurity right.',
            atol=atol,
            rtol=rtol,
        )

    # Run test 2d
    with stopwatch(f'Testing 2D splitter ({splitter2.__class__.__name__})...'):
        if only_1d:
            result2 = test_splitter(
                splitter2, x, y,
                start=start1d,
                end=end1d,
                verbose=params['verbose'],
            )
        else:
            result2 = test_splitter_nd(
                splitter2, XX, Y,
                start=start2d,
                end=end2d,
                verbose=params['verbose'],
            )
        print('Best split found:')
        pprint(result2)

    if multioutput_1d:
        y_sort = y_[sorted_indices]
        result1['feature'] += ax * XX[0].shape[1]
        parent_impurity = 0.5 * (y_sort.var(1).mean() + y_sort.var(0).mean())

        other_axis_imp_left = y_sort[:pos].var(1).mean()
        other_axis_imp_right = y_sort[pos:].var(1).mean()

        result1['impurity_left'] += other_axis_imp_left
        result1['impurity_left'] /= 2
        result1['impurity_right'] += other_axis_imp_right
        result1['impurity_right'] /= 2
        result1['improvement'] = parent_impurity - (
            pos * result1['impurity_left']
            + (y_.shape[0]-pos) * result1['impurity_right']
        ) / y_.shape[0]

        print('GMO-specific updated target impurities:\n'
              '* Left:  {impurity_left}\n'
              '* Right: {impurity_right}\n'
              '* Improvement: {improvement}'.format(**result1))

    # Sometimes a diferent feature yields the same partitioning
    # (especially with few samples in the dataset).
    # assert result2['feature'] == result1['feature'], 'feature differs.'

    assert_allclose(result2['threshold'], result1['threshold'],
                    err_msg='threshold differs from reference.')
    assert_allclose(result2['impurity_left'], result1['impurity_left'],
                    err_msg='impurity_left differs from reference.')
    assert_allclose(result2['impurity_right'], result1['impurity_right'],
                    err_msg='impurity_right differs from reference.')
    assert_allclose(result2['improvement'], result1['improvement'],
                    err_msg='improvement differs from reference.')

    return result1, result2


def test_1d2d_ideal(**params):
    params = DEF_PARAMS | params
    return compare_splitters_1d2d_ideal(
        splitter1=BestSplitter,
        splitter2=BestSplitter,
        **params,
    )


@pytest.mark.parametrize(
    'criterion,bipartite_criterion_adapter',
    [
        (MSE, MSE_Wrapper2D),
        # (MAE, RegressionCriterionWrapper2D),
        (FriedmanMSE, FriedmanAdapter),
        # (Poisson, RegressionCriterionWrapper2D),
    ]
)
def test_1d2d(
    random_state,
    criterion,
    bipartite_criterion_adapter,
    **params
):
    params = DEF_PARAMS | params

    X, Y, x, y, tree = make_interaction_regression(
        return_molten=True,
        return_tree=True,
        n_features=params['nattrs'],
        n_samples=params['shape'],
        random_state=check_random_state(random_state),
    )
    y = y.reshape((-1, 1))
    x = x.astype(np.float32)

    splitter1 = BestSplitter(
        criterion=criterion(n_outputs=y.shape[1], n_samples=x.shape[0]),
        max_features=x.shape[1],
        min_samples_leaf=1,
        min_weight_leaf=0.0,
        random_state=check_random_state(random_state),
    )
    splitter2 = make_2d_splitter(
        splitters=BestSplitter,
        criteria=criterion,
        max_features=params['nattrs'],
        n_samples=Y.shape,
        n_outputs=1,
        random_state=check_random_state(random_state),
        criterion_wrapper_class=bipartite_criterion_adapter,
    )
    result1 = test_splitter(
        splitter1,
        x,
        y,
        verbose=True,
    )
    result2 = test_splitter_nd(
        splitter2,
        X,
        Y,
        verbose=True,
        # verbose=params['verbose'],
    )
    manual_result = manual_split_eval_mse(
        x, y, pos=result1['pos'], feature=result1['feature'],
        start=0, end=y.shape[0],
    )
    assert_equal_dicts(
        result1, manual_result,
        ignore=['improvement'] if criterion == FriedmanMSE else None,
        msg_prefix='(manual) ',
    )

    n_rows, n_cols = Y.shape
    result1['axis'] = int(result1['feature'] >= X[0].shape[1])
    pos = result1['pos']
    result1['pos'] = pos//n_rows if result1['axis'] == 1 else pos//n_cols

    assert_equal_dicts(result1, result2)


def test_ss_1d2d_sup(**params):
    params = DEF_PARAMS | params
    ss2d_splitter = make_2dss_splitter(
        splitters=BestSplitter,
        ss_criteria=SSCompositeCriterion,
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        supervision=1.0,
        max_features=params['nattrs'],
        n_features=params['nattrs'],
        n_samples=params['shape'],
        n_outputs=1,
        random_state=check_random_state(params['seed']),
    )

    return compare_splitters_1d2d(
        splitter1=BestSplitter,
        splitter2=ss2d_splitter,
        **params,
    )


def test_ss_1d2d_unsup(**params):
    params = DEF_PARAMS | params
    ss2d_splitter = make_2dss_splitter(
        splitters=BestSplitter,
        ss_criteria=SSCompositeCriterion,
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        supervision=0.,
        max_features=params['nattrs'],
        n_features=params['nattrs'],
        n_samples=params['shape'],
        n_outputs=1,
    )

    return compare_splitters_1d2d(
        splitter1=BestSplitter,
        splitter2=ss2d_splitter,
        unsupervised_1d=True,
        **params,
    )


def test_ss_1d2d(supervision, **params):
    """Compare 1D to 2D version of semisupervised MSE splitter.
    """
    params = DEF_PARAMS | params

    splitter1 = BestSplitter(
        criterion=make_semisupervised_criterion(
            supervision=supervision,
            supervised_criterion=MSE,
            unsupervised_criterion=MSE,
            n_features=np.sum(params['nattrs']),
            n_samples=np.prod(params['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(params['nattrs']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=check_random_state(params['seed']),
    )

    ss2d_splitter = make_2dss_splitter(
        splitters=BestSplitter,
        ss_criteria=SSCompositeCriterion,
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        supervision=supervision,
        max_features=params['nattrs'],
        n_features=params['nattrs'],
        n_samples=params['shape'],
        n_outputs=1,
        random_state=params['seed'],
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
    )

    return compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=ss2d_splitter,
        semisupervised_1d=True,
        **params,
    )


def test_ss_1d2d_ideal_split(**params):
    params = DEF_PARAMS | params
    ss2d_splitter = make_2dss_splitter(
        splitters=BestSplitter,
        ss_criteria=SSCompositeCriterion,
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        supervision=1.,
        max_features=params['nattrs'],
        n_features=params['nattrs'],
        n_samples=params['shape'],
        n_outputs=1,
    )

    return compare_splitters_1d2d_ideal(
        splitter1=BestSplitter,
        splitter2=ss2d_splitter,
        **params,
    )


def test_ss_1d2d_blobs(supervision, random_state, **params):
    """Test axis decision-semisupervision

    Assert that axis decision semisupervision chooses the split based only on
    supervised data, integrating unsupervised score after it is found.
    """
    params = DEF_PARAMS | params
    n_row_features = params['nattrs'][0]

    X, Y, x, y = make_interaction_blobs(
        return_molten=True,
        n_features=params['nattrs'],
        n_samples=params['shape'],
        random_state=check_random_state(random_state),
        noise=2.0,
        centers=10,
    )

    splitter1 = BestSplitter(
        criterion=make_semisupervised_criterion(
            supervised_criterion=MSE,
            unsupervised_criterion=MSE,
            n_samples=y.shape[0],
            n_outputs=y.shape[1],
            n_features=x.shape[1],
            supervision=supervision,
        ),
        max_features=x.shape[1],
        min_samples_leaf=1,
        min_weight_leaf=0.0,
        random_state=check_random_state(random_state),
    )
    splitter2 = make_2dss_splitter(
        splitters=BestSplitter,
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        supervision=supervision,
        max_features=params['nattrs'],
        n_features=params['nattrs'],
        n_samples=params['shape'],
        n_outputs=1,
        min_samples_leaf=1,
        min_weight_leaf=0.0,
        random_state=check_random_state(random_state),
        axis_decision_only=False,
    )
    result1 = test_splitter(
        splitter1,
        x,
        np.hstack((x, y)),
        verbose=params['verbose'],
    )
    manual_result = manual_split_eval_mse(
        x, y,
        pos=result1['pos'],
        feature=result1['feature'],
        supervision=supervision,
    )
    # rtol=1e-4 because x is converted from float32
    assert_equal_dicts(
        result1, manual_result, msg_prefix='(manual) ', rtol=1e-4,
    )

    n_rows, n_cols = Y.shape
    axis = result1['feature'] >= n_row_features
    result1['axis'] = int(axis)
    pos = result1['pos']
    result1['old_pos'] = pos
    result1['pos'] = pos // n_rows if axis else pos // n_cols

    result2 = test_splitter_nd(
        splitter2,
        X,
        Y,
        verbose=params['verbose'],
    )
    assert_equal_dicts(result2, result1, rtol=1e-4)


@pytest.mark.skip
def test_sfss_1d_sup(**params):
    params = DEF_PARAMS | params

    splitter1 = BestSplitterSFSS(
        criterion=SingleFeatureSSCompositeCriterion(
            supervision=1.,
            criterion=MSE,
            n_features=np.sum(params['nattrs']),
            n_samples=np.prod(params['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(params['nattrs']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=params['seed'],
    )

    return compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=BestSplitter,
        **params,
    )


@pytest.mark.skip
def test_sfss_1d_unsup(**params):
    params = DEF_PARAMS | params
    rstate = check_random_state(params['seed'])

    splitter1 = BestSplitterSFSS(
        criterion=SingleFeatureSSCompositeCriterion(
            supervision=0.,
            criterion=MSE,
            n_features=np.sum(params['nattrs']),
            n_samples=np.prod(params['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(params['nattrs']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=rstate,
    )

    return compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=BestSplitter,
        single_feature_ss_1d=True,
        only_1d=True,
        **params,
    )


@pytest.mark.skip
def test_sfss_2d_sup(**params):
    params = DEF_PARAMS | params
    rstate = check_random_state(params['seed'])

    ss2d_splitter = make_2dss_splitter(
        splitters=BestSplitterSFSS,
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        ss_criteria=SingleFeatureSSCompositeCriterion,
        supervision=1.,
        max_features=params['nattrs'],
        n_features=1,
        n_samples=params['shape'],
        n_outputs=1,
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
        random_state=rstate,
    )

    return compare_splitters_1d2d(
        splitter1=BestSplitter,
        splitter2=ss2d_splitter,
        **params,
    )


@pytest.mark.skip
def test_sfss_2d_unsup(**params):
    params = DEF_PARAMS | params
    rstate = check_random_state(params['seed'])

    ss2d_splitter = make_2dss_splitter(
        splitters=BestSplitterSFSS,
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        ss_criteria=SingleFeatureSSCompositeCriterion,
        supervision=0.,
        max_features=params['nattrs'],
        n_features=1,
        n_samples=params['shape'],
        n_outputs=1,
        random_state=rstate,
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
    )

    return compare_splitters_1d2d(
        splitter1=BestSplitter,
        splitter2=ss2d_splitter,
        single_feature_ss_1d=True,
        unsupervised_1d=True,
        **params,
    )


@pytest.mark.skip
def test_sfss_1d2d(**params):
    """Compare 1D to 2D version of semisupervised MSE splitter.
    """
    params = DEF_PARAMS | params
    rstate = check_random_state(params['seed'])
    supervision = params.get('supervision', -1.)
    if supervision == -1.:
        supervision = rstate.random()

    splitter1 = BestSplitterSFSS(
        criterion=SingleFeatureSSCompositeCriterion(
            supervision=supervision,
            criterion=MSE,
            n_features=1,
            n_samples=np.prod(params['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(params['nattrs']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=check_random_state(params['seed']),
    )

    ss2d_splitter = make_2dss_splitter(
        splitters=BestSplitterSFSS,
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        ss_criteria=SingleFeatureSSCompositeCriterion,
        supervision=supervision,
        max_features=params['nattrs'],
        n_features=1,
        n_samples=params['shape'],
        n_outputs=1,
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
        random_state=check_random_state(params['seed']),
    )

    return compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=ss2d_splitter,
        semisupervised_1d=True,
        **params,
    )


@pytest.mark.skip
def test_ud3_1d2d_unsup(**params):
    """Compare 1D to 2D version of semisupervised MSE splitter.
    """
    params = DEF_PARAMS | params

    splitter1 = BestSplitterSFSS(
        criterion=SingleFeatureSSCompositeCriterion(
            supervision=0.,
            supervised_criterion=MSE,
            unsupervised_criterion=UD3,
            n_features=1,
            n_samples=np.prod(params['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(params['nattrs']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=check_random_state(params['seed']),
    )

    ss2d_splitter = make_2dss_splitter(
        splitters=BestSplitterSFSS,
        ss_criteria=SingleFeatureSSCompositeCriterion,
        supervised_criteria=MSE,
        unsupervised_criteria=UD3,
        supervision=0.,
        max_features=params['nattrs'],
        n_features=1,
        n_samples=params['shape'],
        n_outputs=1,
        random_state=check_random_state(params['seed']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
    )

    result1, result2 = compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=ss2d_splitter,
        # unsupervised_1d=True,
        semisupervised_1d=True,
        manual_impurity=False,
        **params,
    )

    assert result1['feature'] == 0
    assert result2['feature'] == 0


@pytest.mark.skip
def test_ud35_1d2d_unsup(**params):
    """Compare 1D to 2D version of semisupervised MSE splitter.
    """
    params = DEF_PARAMS | params

    splitter1 = BestSplitterSFSS(
        criterion=SingleFeatureSSCompositeCriterion(
            supervision=0.,
            supervised_criterion=MSE,
            unsupervised_criterion=UD35,
            n_features=1,
            n_samples=np.prod(params['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(params['nattrs']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=check_random_state(params['seed']),
    )

    ss2d_splitter = make_2dss_splitter(
        splitters=BestSplitterSFSS,
        ss_criteria=SingleFeatureSSCompositeCriterion,
        supervised_criteria=MSE,
        unsupervised_criteria=UD35,
        supervision=0.,
        max_features=params['nattrs'],
        n_features=1,
        n_samples=params['shape'],
        n_outputs=1,
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
        random_state=check_random_state(params['seed']),
    )

    result1, result2 = compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=ss2d_splitter,
        # unsupervised_1d=True,
        semisupervised_1d=True,
        manual_impurity=False,
        **params,
    )

    assert result1['feature'] == 0
    assert result2['feature'] == 0


@pytest.mark.skip
def test_sfssmse_1d(**params):
    """Compare 1D to 2D version of semisupervised MSE splitter.
    """
    params = DEF_PARAMS | params
    rstate = check_random_state(params['seed'])
    supervision = params.get('supervision', -1.)
    if supervision == -1.:
        supervision = rstate.random()

    splitter1 = BestSplitterSFSS(
        criterion=SFSSMSE(
            supervision=supervision,
            n_features=1,
            n_samples=np.prod(params['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(params['nattrs']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=check_random_state(rstate),
    )

    ss2d_splitter = make_2dss_splitter(
        splitters=BestSplitterSFSS,
        ss_criteria=SingleFeatureSSCompositeCriterion,
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        supervision=supervision,
        max_features=params['nattrs'],
        n_features=1,
        n_samples=params['shape'],
        n_outputs=1,
        random_state=check_random_state(params['seed']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
    )

    return compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=ss2d_splitter,
        semisupervised_1d=True,
        **params,
    )


@pytest.mark.skip
def test_sfssmse_1d2d(**params):
    """Compare 1D to 2D version of semisupervised MSE splitter.
    """
    params = DEF_PARAMS | params
    rstate = np.random.RandomState(params['seed'])
    supervision = params.get('supervision', -1.)
    if supervision == -1.:
        supervision = rstate.random()

    splitter1 = BestSplitterSFSS(
        criterion=SFSSMSE(
            supervision=supervision,
            n_features=1,
            n_samples=np.prod(params['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(params['nattrs']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=check_random_state(params['seed']),
    )

    ss2d_splitter = make_2dss_splitter(
        splitters=BestSplitterSFSS,
        ss_criteria=SFSSMSE,
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        supervision=supervision,
        max_features=params['nattrs'],
        n_features=1,
        n_samples=params['shape'],
        n_outputs=1,
        random_state=check_random_state(params['seed']),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
    )

    return compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=ss2d_splitter,
        semisupervised_1d=True,
        **params,
    )


def test_pbct_splitter(**params):
    params = DEF_PARAMS | params
    splitter2d = make_2d_splitter(
        criterion_wrapper_class=PBCTCriterionWrapper,
        splitters=BestSplitter,
        criteria=AxisMSE,
        max_features=params['nattrs'],
        n_samples=params['shape'],
        n_outputs=params['shape'][::-1],
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.0,
    )

    compare_splitters_1d2d(
        splitter1=BestSplitter,
        splitter2=splitter2d,
        multioutput_1d=True,
        **params,
    )


def test_ss_axis_decision_only(supervision, random_state, **params):
    """Test axis decision-semisupervision

    Assert that axis decision semisupervision chooses the split based only on
    supervised data, integrating unsupervised score after it is found.
    """
    if supervision == 0.0:
        pytest.skip(
            'axis decision only should be compared to supervised splitter'
        )

    params = DEF_PARAMS | params
    n_row_features = params['nattrs'][0]

    X, Y, x, y, gen_tree = make_interaction_regression(
        return_molten=True,
        return_tree=True,
        n_features=params['nattrs'],
        n_samples=params['shape'],
        random_state=check_random_state(random_state),
        max_depth=1,
        max_target=100,
        noise=0.1,
    )
    y = y.reshape((-1, 1))
    x = x.astype('float32')

    splitter2 = make_2dss_splitter(
        splitters=BestSplitter,
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        supervision=supervision,
        max_features=params['nattrs'],
        n_features=params['nattrs'],
        n_samples=params['shape'],
        n_outputs=1,
        random_state=check_random_state(random_state),
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.0,
        axis_decision_only=True,
    )
    result2 = test_splitter_nd(
        splitter2,
        X,
        Y,
        verbose=params['verbose'],
    )

    if result2['axis'] == 0:
        x = x[:, :n_row_features]
    else:  # axis == 1
        x = x[:, n_row_features:]

    splitter1 = BestSplitter(
        criterion=MSE(n_outputs=y.shape[1], n_samples=x.shape[0]),
        max_features=x.shape[1],
        min_samples_leaf=1,
        min_weight_leaf=0.0,
        random_state=check_random_state(random_state),
    )
    result1 = test_splitter(
        splitter1,
        x,
        y,
        verbose=params['verbose'],
    )
    manual_result = manual_split_eval_mse(
        x, y, pos=result1['pos'], feature=result1['feature'],
    )
    assert_equal_dicts(
        result1,
        manual_result,
        ignore={'threshold', 'feature'},
        msg_prefix='(reference) ',
        rtol=1e-4,
        atol=1e-8,
    )
    n_rows, n_cols = params['shape']
    result1['axis'] = result2['axis']
    pos = result1['pos']
    result1['pos'] = pos // n_rows if result1['axis'] else pos // n_cols

    if result1['axis'] == 1:
        result1['feature'] += n_row_features

    # Compare only feature and threshold
    assert_equal_dicts(
        result1,
        result2,
        ignore={
            'impurity_parent',
            'impurity_left',
            'impurity_right',
            'improvement',
        },
        rtol=1e-4,
        atol=1e-8,
    )

