# TODO: test with different axis supervisions. sscrit2d.impurity_improvement()
#       may fail.
import logging
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
    GMO,
    GMOSA,
    SquaredErrorGSO,
    FriedmanGSO,
    RegressionCriterionGSO,
)
from hypertrees.tree._axis_criterion import AxisMSE, AxisFriedmanMSE, AxisGini
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
from test_criteria import mse_impurity, global_mse_impurity


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


@pytest.fixture
def verbose():
    return True


@pytest.fixture(params=[(0, 0, 0, 0), (5, 5, -5, -5)])
def dataset_slice(request):
    # start_row, start_col, end_row, end_col
    return request.param


@pytest.fixture(params=range(10))
def random_state(request):
    return check_random_state(request.param)


@pytest.fixture
def dataset(random_state):
    XX, Y, x, y = make_interaction_regression(
        n_samples=(50, 40),
        n_features=(10, 9),
        n_targets=None,
        min_target=0.0,
        max_target=100.0,
        noise=10.0,
        return_molten=True,
        return_tree=False,
        random_state=random_state,
        max_depth=None,
    )
    return XX, Y, x, y


@pytest.fixture
def dataset_mono_semisupervised(dataset, supervision):
    XX, Y, x, y = dataset
    return XX, Y, x, np.hstack((x, y))


# since we need to convert X from float32 to float64, the actual precision for
# unsupervised data is 1e-4, so that smaller multiples will yield differencees.
@pytest.fixture(params=[0.0, 0.001, 0.1, 0.2328, 0.569, 0.782, 0.995, 1.0])
def supervision(request):
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


def monopartite_splitter_factory(
    request,
    x,
    y,
    *,
    splitter,
    criterion,
):
    """Returns a factory to make a splitter based on the dataset provided.
    """
    return splitter(
        criterion=criterion(
            n_outputs=y.shape[1],
            n_samples=x.shape[0],
        ),
        max_features=x.shape[1],
        min_samples_leaf=1,
        min_weight_leaf=0.0,
        random_state=request.getfixturevalue('random_state'),
    )


def monopartite_ss_splitter_factory(
    request,
    x,
    y,
    *,
    criterion,
    unsupervised_criterion,
    splitter,
    ss_adapter=None,
):
    result = splitter(
        criterion=make_semisupervised_criterion(
            supervision=supervision,
            supervised_criterion=criterion,
            unsupervised_criterion=unsupervised_criterion,
            n_features=x.shape[1],
            n_samples=x.shape[0],
            n_outputs=y.shape[1],
        ),
        max_features=x.shape[1],
        min_samples_leaf=1,
        min_weight_leaf=0.0,
        ss_adapter=ss_adapter,
        random_state=request.getfixturevalue('random_state'),
    )
    result.criterion.init_X(x)


def bipartite_splitter_factory(
    request,
    x, y,
    criterion_wrapper_class,
    *args,
    **kwargs,
):
    return make_2d_splitter(
        *args,
        max_features=[X.shape[1] for X in x],
        n_samples=y.shape,
        n_outputs=np.sum(y.shape) if criterion_wrapper_class == GMO else 1,
        random_state=request.getfixturevalue('random_state'),
        criterion_wrapper_class=criterion_wrapper_class,
        **kwargs
    )


@pytest.mark.parametrize(
    (
        'splitter1_factory, splitter1_args,'
        'splitter2_factory, splitter2_args'
    ), [
        (
            monopartite_splitter_factory, {
                'splitter': BestSplitter,
                'criterion': MSE,
            },
            monopartite_splitter_factory, {  # TODO: semisuper
                'splitter': BestSplitter,
                'criterion': MSE,
            },
        ),
    ]
)
def test_compare_1d_splitters(
    request,
    splitter1_factory,
    splitter1_args,
    splitter2_factory,
    splitter2_args,
    dataset,
    dataset_slice,
):
    XX, Y, *_ = dataset
    X = XX[0]
    start, _, end, _ = dataset_slice
    splitter1 = splitter1_factory(request, X, Y, **splitter1_args)
    splitter2 = splitter2_factory(request, X, Y, **splitter2_args)

    with stopwatch(
        f'Testing first splitter ({splitter1.__class__.__name__})...'
    ):
        result1 = test_splitter(
            splitter1, X, Y,
            start=start,
            end=end,
            verbose=verbose,
        )
    print('Best split 1 found:')
    pprint(result1)

    with stopwatch(
        f'Testing second splitter ({splitter2.__class__.__name__})...'
    ):
        result2 = test_splitter(
            splitter2, X, Y,
            start=start,
            end=end,
            verbose=verbose,
        )
    print('Best split 2 found:')
    pprint(result2)

    assert_equal_dicts(result1, result2)


@pytest.mark.parametrize(
    (
        'mono_splitter_factory, mono_splitter_args,'
        'bi_splitter_factory, bi_splitter_args'
    ), [
        (
            monopartite_splitter_factory, {
                'splitter': BestSplitter,
                'criterion': MSE,
            },
            bipartite_splitter_factory, {
                'splitters': BestSplitter,
                'criteria': MSE,
                'criterion_wrapper_class': SquaredErrorGSO,
            },
        ),
        (
            monopartite_splitter_factory, {
                'splitter': BestSplitter,
                'criterion': FriedmanMSE,
            },
            bipartite_splitter_factory, {
                'splitters': BestSplitter,
                'criteria': FriedmanMSE,
                'criterion_wrapper_class': FriedmanGSO,
            },
        ),
    ],
    ids=['mse', 'friedman'],
)
def test_compare_1d2d_splitters_gso(
    request,
    mono_splitter_factory,
    mono_splitter_args,
    bi_splitter_factory,
    bi_splitter_args,
    dataset,
    dataset_slice,
):
    XX, Y, x, y = dataset
    start_row, start_col, end_row, end_col = dataset_slice
    # Columns subset not supported.
    start_col, end_col = 0, Y.shape[1]

    mono_splitter = mono_splitter_factory(request, x, y, **mono_splitter_args)
    bi_splitter = bi_splitter_factory(request, XX, Y, **bi_splitter_args)

    with stopwatch(
        f'Testing monopartite splitter ({mono_splitter.__class__.__name__})...'
    ):
        result_monopartite = test_splitter(
            mono_splitter,
            x, y,
            start=start_row * Y.shape[1],
            end=end_row * Y.shape[1],
            verbose=verbose,
        )
    print('Best monopartite split found:')
    pprint(result_monopartite)

    # Adapt values to compare
    n_rows, n_cols = Y.shape
    result_monopartite['axis'] = (
        int(result_monopartite['feature'] >= XX[0].shape[1])
    )
    pos = result_monopartite['pos']

    # FIXME: It's complicated whem we use start_row != 0
    result_monopartite['pos'] = (
        pos // n_rows if result_monopartite['axis'] == 1 else pos // n_cols
    )

    print('Corrected monopartite split:')
    pprint(result_monopartite)

    with stopwatch(
        f'Testing bipartite splitter ({bi_splitter.__class__.__name__})...'
    ):
        result_bipartite = test_splitter_nd(
            bi_splitter,
            XX, Y,
            start=[start_row, start_col],
            end=[end_row, end_col],
            verbose=verbose,
        )
    print('Best bipartite split found:')
    pprint(result_bipartite)

    # NOTE: Sometimes a diferent feature yields the same partitioning
    # (especially with few samples in the dataset).
    assert_equal_dicts(
        result_monopartite,
        result_bipartite,
        ignore=['pos'], # Complicated if start_row != 0. Threshold is enough.
    )


@pytest.mark.parametrize(
    (
        'mono_splitter_factory, mono_splitter_args,'
        'bi_splitter_factory, bi_splitter_args'
    ), [
        (
            monopartite_splitter_factory, {
                'splitter': BestSplitter,
                'criterion': MSE,
            },
            bipartite_splitter_factory, {
                'splitters': BestSplitter,
                'criteria': AxisMSE,
                'criterion_wrapper_class': GMO,
            },
        ),
        (
            monopartite_splitter_factory, {
                'splitter': BestSplitter,
                'criterion': FriedmanMSE,
            },
            bipartite_splitter_factory, {
                'splitters': BestSplitter,
                'criteria': AxisFriedmanMSE,
                'criterion_wrapper_class': GMO,
            },
        ),
    ],
    ids=['mse', 'friedman'],
)
def test_compare_1d2d_splitters_gmo(
    request,
    mono_splitter_factory,
    mono_splitter_args,
    bi_splitter_factory,
    bi_splitter_args,
    dataset,
    dataset_slice,
):
    XX, Y, x, y = dataset
    start_row, start_col, end_row, end_col = dataset_slice

    mono_splitter = mono_splitter_factory(request, x, y, **mono_splitter_args)
    bi_splitter = bi_splitter_factory(request, XX, Y, **bi_splitter_args)

    with stopwatch(
        f'Testing 2D splitter ({bi_splitter.__class__.__name__})...'
    ):
        result_bipartite = test_splitter_nd(
            bi_splitter,
            XX, Y,
            start=[start_row, start_col],
            end=[end_row, end_col],
            verbose=verbose,
        )
    print('Best bipartite split found:')
    pprint(result_bipartite)

    with stopwatch(
        f'Testing 1D splitter ({mono_splitter.__class__.__name__}) on rows...'
    ):
        result_rows = test_splitter(
            mono_splitter,
            XX[0], Y,
            start=start_row,
            end=end_row,
            verbose=verbose,
        )
    result_rows['axis'] = 0
    print('Best monopartite rows split found:')
    pprint(result_rows)

    with stopwatch(
        f'Testing 1D splitter ({mono_splitter.__class__.__name__}) on columns...'
    ):
        result_columns = test_splitter(
            mono_splitter,
            XX[1], np.ascontiguousarray(Y.T),
            start=start_col,
            end=end_col,
            verbose=verbose,
        )
    result_columns['axis'] = 1
    result_columns['feature'] += XX[0].shape[1]
    print('Best monopartite columns split found:')
    pprint(result_columns)

    results_monopartite = (result_rows, result_columns)
    best_axis = result_bipartite['axis']

    assert (
        results_monopartite[best_axis]['improvement'] >
        results_monopartite[1 - best_axis]['improvement']
    ), 'Wrong axis.'

    assert_equal_dicts(
        results_monopartite[best_axis],
        result_bipartite,
        ignore=['axis'],
    )


@pytest.mark.skip
def test_1d2d_ideal(**params):
    params = DEF_PARAMS | params
    return compare_splitters_1d2d_ideal(
        splitter1=BestSplitter,
        splitter2=BestSplitter,
        **params,
    )


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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
        noise=5.0,
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


@pytest.mark.skip
def test_splitter_gmo(**params):
    params = DEF_PARAMS | params
    BipartiteSplitter = make_2d_splitter(
        max_features=params['nattrs'],
        n_samples=params['shape'],
        n_outputs=params['shape'][::-1],
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.0,
    )

    compare_splitters_1d2d_ideal(
        splitter1=BestSplitter,
        splitter2=BipartiteSplitter,
        multioutput_1d=True,
        **params,
    )


# TODO: compare results
@pytest.mark.skip
def test_splitter_gmo_classification(**params):
    params = DEF_PARAMS | params
    BipartiteSplitter = make_2d_splitter(
        criterion_wrapper_class=GMO,
        is_classification=True,
        splitters=BestSplitter,
        criteria=[AxisGini, AxisGini],
        max_features=params['nattrs'],
        n_samples=params['shape'],
        n_outputs=params['shape'][::-1],
        n_classes=[np.repeat(2, i) for i in params['shape'][::-1]],
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.0,
    )

    with pytest.raises(AssertionError):
        compare_splitters_1d2d(
            splitter1=BestSplitter,
            splitter2=BipartiteSplitter,
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
    splitter2.criterion_wrapper.init_X(X[0], X[1])
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


@pytest.mark.parametrize('axis_decision_only', (True, False))
def test_criterion_identity_in_wrappers(axis_decision_only):
    # splitter1 = BestSplitter(
    #     criterion=make_semisupervised_criterion(
    #         supervised_criterion=MSE,
    #         unsupervised_criterion=MSE,
    #         n_samples=10,
    #         n_outputs=10,
    #         n_features=10,
    #         supervision=supervision,
    #     ),
    #     max_features=10,
    #     min_samples_leaf=1,
    #     min_weight_leaf=0.0,
    #     random_state=check_random_state(random_state),
    # )
    splitter = make_2dss_splitter(
        splitters=BestSplitter,
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        supervision=0.1,
        max_features=10,
        n_features=10,
        n_samples=10,
        n_outputs=1,
        min_samples_leaf=1,
        min_weight_leaf=0.0,
        random_state=0,
        axis_decision_only=axis_decision_only,
    )
    assert (
        splitter.criterion_wrapper.supervised_bipartite_criterion.criterion_rows
        is splitter.criterion_wrapper.supervised_criterion_rows
    )
    assert (
        splitter.criterion_wrapper.supervised_bipartite_criterion.criterion_cols
        is splitter.criterion_wrapper.supervised_criterion_cols
    )

    if axis_decision_only:
        assert (
            splitter.splitter_rows.criterion
            is splitter.criterion_wrapper.supervised_criterion_rows
        )
        assert (
            splitter.splitter_cols.criterion
            is splitter.criterion_wrapper.supervised_criterion_cols
        )

    else:
        assert (
            splitter.splitter_rows.criterion
            is splitter.criterion_wrapper.ss_criterion_rows
        )
        assert (
            splitter.splitter_cols.criterion
            is splitter.criterion_wrapper.ss_criterion_cols
        )
        assert (
            splitter.splitter_rows.criterion.supervised_criterion
            is splitter.criterion_wrapper.supervised_criterion_rows
        )
        assert (
            splitter.splitter_rows.criterion.unsupervised_criterion
            is splitter.criterion_wrapper.unsupervised_criterion_rows
        )
        assert (
            splitter.splitter_cols.criterion.supervised_criterion
            is splitter.criterion_wrapper.supervised_criterion_cols
        )
        assert (
            splitter.splitter_cols.criterion.unsupervised_criterion
            is splitter.criterion_wrapper.unsupervised_criterion_cols
        )


def test_gini_mse_identity(random_state, **params):
    params = DEF_PARAMS | params
    XX, Y = make_interaction_regression(
        n_samples=(40, 50),
        n_features=(9, 10),
        random_state=random_state,
    )

    # XX = [check_symmetric(X, raise_warning=False) for X in XX]
    Y = (Y > Y.mean()).astype('float64')  # Turn into binary.

    splitter_gini = make_2d_splitter(
        criterion_wrapper_class=GMO,
        splitters=BestSplitter,
        criteria=AxisGini,
        max_features=params['nattrs'],
        n_classes=[np.repeat(2, n) for n in params['shape'][::-1]],
        n_outputs=params['shape'][::-1],
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.0,
    )
    splitter_mse = make_2d_splitter(
        criterion_wrapper_class=GMO,
        splitters=BestSplitter,
        criteria=AxisMSE,
        max_features=params['nattrs'],
        n_samples=params['shape'],
        n_outputs=params['shape'][::-1],
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.0,
    )
    result_mse = test_splitter_nd(
        splitter_mse, XX, Y,
        verbose=params['verbose'],
    )
    result_gini = test_splitter_nd(
        splitter_gini, XX, Y,
        verbose=params['verbose'],
    )

    result_gini['original_improvement'] = result_gini['improvement']
    result_gini['improvement'] /= 2

    assert_equal_dicts(
        result_gini,
        result_mse,
        ignore={
            'impurity_parent',
            'impurity_left',
            'impurity_right',
            'original_improvement',
        },
    )
