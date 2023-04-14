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

from bipartite_learn.tree._splitter_factory import (
    make_bipartite_splitter,
    make_bipartite_ss_splitter,
    make_semisupervised_criterion,
)
from bipartite_learn.tree._bipartite_criterion import (
    GMO,
    GMOSA,
)
from bipartite_learn.tree._axis_criterion import (
    AxisSquaredError,
    AxisSquaredErrorGSO,
    AxisFriedmanGSO,
    AxisGini,
    AxisEntropy,
)
from bipartite_learn.tree._semisupervised_criterion import (
    SSCompositeCriterion,
)
from bipartite_learn.tree._unsupervised_criterion import (
    UnsupervisedSquaredError,
    UnsupervisedFriedman,
)
from .utils.make_examples import make_interaction_blobs, make_interaction_regression

from .utils.tree_utils import apply_monopartite_splitter, apply_bipartite_splitter
from .utils.test_utils import (
    parse_args, stopwatch, gen_mock_data, melt_2d_data, assert_equal_dicts,
)
from . import test_criteria


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


@pytest.fixture(params=[(0, 0, 0, 0)])#, (5, 5, -5, -5)])
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


@pytest.fixture(params=[0.0, 0.001, 0.1, 0.2328, 0.569, 0.782, 0.995, 1.0])
def supervision(request):
    return request.param


@pytest.fixture(
    params=[
        (0.0, 0.0), (0.1, 0.9), (0.2328, 0.569), (0.782, 0.782), (1.0, 1.0),
    ],
)
def bipartite_supervision(request):
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
    x,
    y,
    *,
    random_state,
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
        random_state=random_state,
    )


def monopartite_ss_splitter_factory(
    x,
    y,
    *,
    random_state,
    splitter,
    supervised_criterion,
    unsupervised_criterion,
    supervision,
    ss_adapter=None,
):

    criterion = make_semisupervised_criterion(
        supervision=supervision,
        supervised_criterion=supervised_criterion,
        unsupervised_criterion=unsupervised_criterion,
        n_features=x.shape[1],
        n_samples=x.shape[0],
        n_outputs=y.shape[1],
        ss_class=ss_adapter,
    )
    criterion.set_X(x.astype('float64'))

    return splitter(
        criterion=criterion,
        max_features=x.shape[1],
        min_samples_leaf=1,
        min_weight_leaf=0.0,
        random_state=random_state,
    )


def bipartite_splitter_factory(
    x, y,
    *,
    random_state,
    **kwargs,
):
    return make_bipartite_splitter(
        max_features=[X.shape[1] for X in x],
        n_samples=y.shape,
        n_outputs=y.shape[::-1],
        random_state=random_state,
        **kwargs,
    )


def bipartite_ss_splitter_factory(
    x, y,
    *,
    random_state,
    **kwargs,
):
    splitter = make_bipartite_ss_splitter(
        max_features=[X.shape[1] for X in x],
        n_samples=y.shape,
        n_outputs=y.shape[::-1],
        n_features=[X.shape[1] for X in x],
        random_state=random_state,
        **kwargs,
    )
    splitter.criterion_wrapper.set_X(
        x[0].astype('float64'),
        x[1].astype('float64'),
    )
    return splitter


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
            monopartite_splitter_factory, {
                'splitter': BestSplitter,
                'criterion': MSE,
            },
        ),
    ],
    ids=['mse_ss_sup'],
)
def test_compare_1d_splitters(
    splitter1_factory,
    splitter1_args,
    splitter2_factory,
    splitter2_args,
    dataset,
    dataset_slice,
    random_state,
):
    XX, Y, *_ = dataset
    X = XX[0]
    start, _, end, _ = dataset_slice

    splitter1 = splitter1_factory(
        X, Y,
        random_state=random_state,
        **splitter1_args,
    )
    splitter2 = splitter2_factory(
        X, Y,
        random_state=random_state,
        **splitter2_args,
    )

    with stopwatch(
        f'Testing first splitter ({splitter1.__class__.__name__})...'
    ):
        result1 = apply_monopartite_splitter(
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
        result2 = apply_monopartite_splitter(
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
                'criteria': AxisSquaredErrorGSO,
                'bipartite_criterion_class': GMOSA,
            },
        ),
        # Friedman impurity depends on weighted_n_left/right, predictable
        # failing here.
    ],
    ids=['mse'],
)
def test_compare_1d2d_splitters_gso(
    mono_splitter_factory,
    mono_splitter_args,
    bi_splitter_factory,
    bi_splitter_args,
    dataset,
    dataset_slice,
    random_state,
):
    XX, Y, x, y = dataset
    start_row, start_col, end_row, end_col = dataset_slice
    # Columns subset not supported.
    start_col, end_col = 0, 0

    mono_splitter = mono_splitter_factory(
        x, y,
        random_state=random_state,
        **mono_splitter_args,
    )
    bi_splitter = bi_splitter_factory(
        XX, Y,
        random_state=random_state,
        **bi_splitter_args,
    )

    with stopwatch(
        f'Testing monopartite splitter ({mono_splitter.__class__.__name__})...'
    ):
        result_monopartite = apply_monopartite_splitter(
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
        result_bipartite = apply_bipartite_splitter(
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
                'criteria': AxisSquaredError,
                'bipartite_criterion_class': GMOSA,
            },
        ),
        (
            monopartite_splitter_factory, {
                'splitter': BestSplitter,
                'criterion': FriedmanMSE,
            },
            bipartite_splitter_factory, {
                'splitters': BestSplitter,
                'criteria': AxisFriedmanGSO,
                'bipartite_criterion_class': GMOSA,
            },
        ),
    ],
    ids=['mse', 'friedman'],
)
def test_compare_1d2d_splitters_gmo(
    mono_splitter_factory,
    mono_splitter_args,
    bi_splitter_factory,
    bi_splitter_args,
    dataset,
    dataset_slice,
    random_state,
):
    XX, Y, x, y = dataset
    start_row, start_col, end_row, end_col = dataset_slice

    mono_splitter_rows = mono_splitter_factory(
        XX[0], Y,
        random_state=random_state,
        **mono_splitter_args,
    )
    mono_splitter_cols = mono_splitter_factory(
        XX[1], Y.T,
        random_state=random_state,
        **mono_splitter_args,
    )

    bi_splitter = bi_splitter_factory(
        XX, Y,
        random_state=random_state,
        **bi_splitter_args,
    )

    with stopwatch(
        f'Testing 2D splitter ({bi_splitter.__class__.__name__})...'
    ):
        result_bipartite = apply_bipartite_splitter(
            bi_splitter,
            XX, Y,
            start=[start_row, start_col],
            end=[end_row, end_col],
            verbose=verbose,
        )
    print('Best bipartite split found:')
    pprint(result_bipartite)

    with stopwatch(
        f'Testing 1D splitter ({type(mono_splitter_rows).__name__}) on rows...'
    ):
        result_rows = apply_monopartite_splitter(
            mono_splitter_rows,
            XX[0], Y,
            start=start_row,
            end=end_row,
            verbose=verbose,
        )
    result_rows['axis'] = 0
    print('Best monopartite rows split found:')
    pprint(result_rows)

    with stopwatch(
        f'Testing 1D splitter ({type(mono_splitter_cols).__name__}) on columns...'
    ):
        result_columns = apply_monopartite_splitter(
            mono_splitter_cols,
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
        subset={'pos', 'feaure', 'threshold'},
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
            monopartite_splitter_factory, {
                'splitter': BestSplitter,
                'criterion': MSE,
            },
        ),
    ],
    ids=['mse'],
)
def test_compare_1d_splitters_ss(
    splitter1_factory,
    splitter1_args,
    splitter2_factory,
    splitter2_args,
    dataset,
    dataset_slice,
    supervision,
    random_state,
):
    if splitter1_args.get('supervision', False) is None:
        splitter1_args['supervision'] = supervision

    if splitter2_args.get('supervision', False) is None:
        splitter2_args['supervision'] = supervision
    
    X, Y, x, y = dataset
    X[0] *= (1 - supervision)
    Y *= supervision
    dataset = X, Y, x, y

    test_compare_1d_splitters(
        splitter1_factory,
        splitter1_args,
        splitter2_factory,
        splitter2_args,
        dataset,
        dataset_slice,
        random_state=random_state,
    )


@pytest.mark.parametrize(
    (
        'mono_splitter_factory, mono_splitter_args,'
        'bi_splitter_factory, bi_splitter_args'
    ), [
        (
            monopartite_ss_splitter_factory, {
                'splitter': BestSplitter,
                'supervised_criterion': UnsupervisedSquaredError,
                'unsupervised_criterion': UnsupervisedSquaredError,
                'supervision': None,
            },
            bipartite_ss_splitter_factory, {
                'splitters': BestSplitter,
                'supervised_criteria': AxisSquaredErrorGSO,
                'unsupervised_criteria': UnsupervisedSquaredError,
                'supervision': None,
            },
        ),
        (
            monopartite_ss_splitter_factory, {
                'splitter': BestSplitter,
                'supervised_criterion': UnsupervisedFriedman,
                'unsupervised_criterion': UnsupervisedSquaredError,
                'supervision': None,
            },
            bipartite_ss_splitter_factory, {
                'splitters': BestSplitter,
                'supervised_criteria': AxisFriedmanGSO,
                'unsupervised_criteria': UnsupervisedSquaredError,
                'supervision': None,
            },
        )
    ],
    ids=['mse', 'friedman'],
)
def test_compare_1d2d_splitters_gso_ss(
    mono_splitter_factory,
    mono_splitter_args,
    bi_splitter_factory,
    bi_splitter_args,
    dataset,
    dataset_slice,
    bipartite_supervision,
    random_state,
):
    if mono_splitter_args.get('supervision', False) is None:
        mono_splitter_args['supervision'] = bipartite_supervision[0]

    if bi_splitter_args.get('supervision', False) is None:
        bi_splitter_args['supervision'] = bipartite_supervision

    test_compare_1d2d_splitters_gso(
        mono_splitter_factory,
        mono_splitter_args,
        bi_splitter_factory,
        bi_splitter_args,
        dataset,
        dataset_slice,
        random_state=random_state,
    )


@pytest.mark.parametrize(
    (
        'mono_splitter_factory, mono_splitter_args,'
        'bi_splitter_factory, bi_splitter_args'
    ), [
        (
            monopartite_ss_splitter_factory, {
                'splitter': BestSplitter,
                'supervised_criterion': UnsupervisedFriedman,
                'unsupervised_criterion': UnsupervisedSquaredError,
                'supervision': None,
            },
            bipartite_ss_splitter_factory, {
                'splitters': BestSplitter,
                'supervised_criteria': AxisFriedmanGSO,
                'unsupervised_criteria': UnsupervisedSquaredError,
                'supervision': None,
            },
        ),
        (
            monopartite_ss_splitter_factory, {
                'splitter': BestSplitter,
                'supervised_criterion': UnsupervisedSquaredError,
                'unsupervised_criterion': UnsupervisedSquaredError,
                'supervision': None,
            },
            bipartite_ss_splitter_factory, {
                'splitters': BestSplitter,
                'supervised_criteria': AxisSquaredError,
                'unsupervised_criteria': UnsupervisedSquaredError,
                'supervision': None,
            },
        ),
    ],
    ids=['friedman', 'mse'],
)
def test_compare_1d2d_splitters_gmo_ss(
    mono_splitter_factory,
    mono_splitter_args,
    bi_splitter_factory,
    bi_splitter_args,
    dataset,
    dataset_slice,
    bipartite_supervision,
    random_state,
):
    if mono_splitter_args.get('supervision', False) is None:
        mono_splitter_args['supervision'] = bipartite_supervision[0]

    if bi_splitter_args.get('supervision', False) is None:
        bi_splitter_args['supervision'] = bipartite_supervision

    test_compare_1d2d_splitters_gmo(
        mono_splitter_factory,
        mono_splitter_args,
        bi_splitter_factory,
        bi_splitter_args,
        dataset,
        dataset_slice,
        random_state=random_state,
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
    ss2d_splitter = make_bipartite_ss_splitter(
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

    ss2d_splitter = make_bipartite_ss_splitter(
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
    ss2d_splitter = make_bipartite_ss_splitter(
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


@pytest.mark.skip
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
    splitter2 = make_bipartite_ss_splitter(
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
    result1 = apply_monopartite_splitter(
        splitter1,
        x,
        np.hstack((x, y)),
        verbose=params['verbose'],
    )
    ref_criterion = test_criteria.ReferenceSquaredError()
    ref_criterion.set_data(x, y, supervision=supervision)

    manual_result = ref_criterion.evaluate_split(
        pos=result1['pos'],
        feature=result1['feature'],
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

    result2 = apply_bipartite_splitter(
        splitter2,
        X,
        Y,
        verbose=params['verbose'],
    )
    assert_equal_dicts(result2, result1, rtol=1e-4)


@pytest.mark.skip
def apply_monopartite_splitter_gmo(**params):
    params = DEF_PARAMS | params
    BipartiteSplitter = make_bipartite_splitter(
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
def apply_monopartite_splitter_gmo_classification(**params):
    params = DEF_PARAMS | params
    BipartiteSplitter = make_bipartite_splitter(
        bipartite_criterion_class=GMO,
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


@pytest.mark.skip
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
        random_state=random_state,
        max_depth=1,
        max_target=100,
        noise=0.1,
    )
    y = y.reshape((-1, 1))

    splitter2 = make_bipartite_ss_splitter(
        splitters=BestSplitter,
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        supervision=supervision,
        max_features=params['nattrs'],
        n_features=params['nattrs'],
        n_samples=params['shape'],
        n_outputs=1,
        random_state=random_state,
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.0,
        axis_decision_only=True,
    )
    splitter2.criterion_wrapper.init_X(X[0], X[1])
    result2 = apply_bipartite_splitter(
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
    result1 = apply_monopartite_splitter(
        splitter1,
        x,
        y,
        verbose=params['verbose'],
    )
    ref_criterion = test_criteria.ReferenceSquaredError()
    ref_criterion.set_data(x, y, supervision=supervision)

    manual_result = ref_criterion.evaluate_split(
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


def test_gini_mse_identity(random_state, **params):
    params = DEF_PARAMS | params
    XX, Y = make_interaction_regression(
        n_samples=(40, 50),
        n_features=(9, 10),
        random_state=random_state,
    )

    # XX = [check_symmetric(X, raise_warning=False) for X in XX]
    Y = (Y > Y.mean()).astype('float64')  # Turn into binary.
    node_impurity = (Y.var(0).mean() + Y.var(1).mean()) / 2
    logging.info(f'parent_impurity: {node_impurity}')

    splitter_gini = make_bipartite_splitter(
        bipartite_criterion_class=GMO,
        splitters=BestSplitter,
        criteria=AxisGini,
        max_features=params['nattrs'],
        n_classes=[np.repeat(2, n) for n in params['shape'][::-1]],
        n_outputs=params['shape'][::-1],
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.0,
    )
    splitter_mse = make_bipartite_splitter(
        bipartite_criterion_class=GMO,
        splitters=BestSplitter,
        criteria=AxisSquaredError,
        max_features=params['nattrs'],
        n_samples=params['shape'],
        n_outputs=params['shape'][::-1],
        min_samples_leaf=params['min_samples_leaf'],
        min_weight_leaf=0.0,
    )
    result_mse = apply_bipartite_splitter(
        splitter_mse, XX, Y,
        verbose=params['verbose'],
    )
    result_gini = apply_bipartite_splitter(
        splitter_gini, XX, Y,
        verbose=params['verbose'],
    )

    result_gini['improvement'] /= 2
    result_gini['impurity_left'] /= 2
    result_gini['impurity_right'] /= 2
    result_gini['impurity_parent'] /= 2

    assert_equal_dicts(result_gini, result_mse)
