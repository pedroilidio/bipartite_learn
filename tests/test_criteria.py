import logging
from typing import Callable, Dict
import numpy as np
import scipy.stats
import pytest
from collections import defaultdict
from numbers import Real, Integral
from sklearn.utils.validation import check_random_state
from sklearn.utils._testing import assert_allclose
from sklearn.utils._param_validation import validate_params, Interval
from sklearn.tree._criterion import MSE, MAE, FriedmanMSE, Poisson
from sklearn.tree._splitter import BestSplitter
from hypertrees.tree._splitter_factory import (
    make_semisupervised_criterion,
    make_bipartite_criterion,
    make_2dss_criterion,
    _infer_ss_adapter_type,
)
from hypertrees.tree._axis_criterion import (
    AxisMSE,
    AxisFriedmanMSE,
    AxisSquaredErrorGSO
)
from hypertrees.tree._nd_criterion import (
    SquaredErrorGSO, FriedmanGSO, GMO, GMOSA,
)
from hypertrees.tree._semisupervised_criterion import (
    SSCompositeCriterion,
    HomogeneousCompositeSS,
    AxisHomogeneousCompositeSS,
)

from hypertrees.melter import row_cartesian_product
from test_utils import assert_equal_dicts, comparison_text
from make_examples import make_interaction_blobs
from splitter_test import (
    apply_criterion,
    apply_bipartite_criterion,
    apply_bipartite_ss_criterion,
    get_criterion_status,
    update_criterion,
)


@pytest.fixture(params=[0.0, 0.00001, 0.1, 0.2328, 0.569, 0.782, 0.995, 1.0])
def supervision(request):
    return request.param


@pytest.fixture(params=range(10))
def random_state(request):
    return check_random_state(request.param)


@pytest.fixture
def n_samples(request):
    return (10, 9)


@pytest.fixture(params=[(3, 3), (2, 3)])
def n_features(request):
    return request.param


@pytest.fixture
def data(n_samples, n_features, random_state):
    X, Y, x, y = make_interaction_blobs(
        return_molten=True,
        n_features=n_features,
        n_samples=n_samples,
        random_state=random_state,
        noise=2.0,
        centers=10,
    )
    return X, Y, x, y


def assert_ranks_are_close(a, b, **kwargs):
    a_s = np.argsort(a)
    b_s = np.argsort(b)
    # If any of swapped values (between both b and a ordering) is not
    # different, do not raise.
    try:
        assert_equal_dicts(
            {'a_sorted': a[a_s], 'b_sorted': b[a_s]},
            {'a_sorted': a[b_s], 'b_sorted': b[b_s]},
            **kwargs,
        )
    except AssertionError as error:
        if "'b_sorted' and 'a_sorted" in str(error):
            raise


# TODO: is it necessary to not simplify?
def mse_impurity(a):
    return a.var(0).mean()


def global_mse_impurity(a):
    return a.var()


def sort_by_feature(x, y, feature):
    sorted_indices = x[:, feature].argsort()
    return x[sorted_indices], y[sorted_indices]


@validate_params({
    'X': ['array-like'],
    'y': ['array-like'],
    'start': [Integral],
    'end': [Integral],
    'feature': [Interval(Integral, 0, None, closed='left'), None],
    'supervision': [Interval(Real, 0.0, 1.0, closed='both'), None],
    'average_both_axes': ['boolean'],
    'impurity': [callable],
})
def evaluate_split(
    X: np.ndarray,
    y: np.ndarray,
    pos: int,
    start: int = 0,
    end: int = 0,
    *,
    feature: int | None = None,
    supervision: float | None = None,
    average_both_axes: bool = False,
    sample_weight: np.ndarray | None = None,
    weighted_n_samples: float | None = None,
    weighted_n_node_samples: float | None = None,
    impurity: Callable[[np.ndarray], float] = mse_impurity,
    unsupervised_impurity: Callable[[np.ndarray], float] | None = None,
) -> Dict[str, int | float]:
    """
    Evaluates the impurities of a given split position in a dataset, simulating
    split evaluation by the Criterion objects of decision trees.

    Parameters
    ----------
    X : np.ndarray
        The input data of shape (n_samples, n_features).
    y : np.ndarray
        The target values of shape (n_samples, n_outputs).
    pos : int
        The split position to evaluate. The absolute index must be provided,
        relative to the begining of the dataset, NOT relative to `start`.
    start : int, optional
        The starting position of the data to evaluate, representing the start
        index of the current node, by default 0.
    end : int, optional
        The ending position of the data to evaluate, representing the final
        index of the current node, by default 0.
    feature : int, optional
        The feature index representing the column of X to sort the data by. If
        None, no sorting is performed. By default None.
    supervision : float between 0 and 1, optional
        If not None, a semisupervised impurity is metric is considered instead,
        calculated as follows:
        ``
            supervision \
                * supervised_impurity(y) / root_supervised_impurity
            + (1 - supervision) \
                * unsupervised_impurity(x) / root_unsupervised_impurity
        ``
        The impurities at the root node are simply taken as to be the current
        node impurity (i.e. the parent impurity). By default None.
    average_both_axes : bool, optional
        Whether to average the impurity on both axes. Useful to evaluate
        bipartite criteria along each axis. By default False.
    sample_weight : np.ndarray, optional
        Sample weights of shape (n_samples, ), by default None.
    weighted_n_samples : float, optional
        The total weighted number of samples in the dataset, taken as
        `sample_weight.sum()` if None, by default None.
    weighted_n_node_samples : float, optional
        The total weighted number of samples in the current node, taken to be
        `sample_weight[start:end].sum()` if None, by default None.
    impurity : Callable[[np.ndarray], float], optional
        The function to use for calculating impurity, by default mse_impurity.
    unsupervised_impurity : Callable[[np.ndarray], float], optional
        The function to use for calculating unsupervised_impurity. If None,
        we set unsupervised_impurity = impurity. By default None.

    Returns
    -------
    Dict[str, int | float]
        A dictionary with the following keys and values:
        - 'pos': the split position.
        - 'impurity_parent': the impurity of the parent node.
        - 'weighted_n_left': the weighted number of samples in the left node.
        - 'weighted_n_right': the weighted number of samples in the right node.
        - 'impurity_left': the impurity of the left node.
        - 'impurity_right': the impurity of the right node.
        - 'improvement': the impurity improvement gained by the split.

        If `supervision` is provided, the following values will be included:
        - 'supervised_impurity_parent': the supervised impurity of the parent node.
        - 'unsupervised_impurity_parent': the unsupervised impurity of the parent node.
        - 'supervised_impurity_left': the supervised impurity of the left node.
        - 'supervised_impurity_right': the supervised impurity of the right node.
        - 'unsupervised_impurity_left': the unsupervised impurity of the left node.
        - 'unsupervised_impurity_right': the unsupervised impurity of the right node.
    """
    start = start if start >= 0 else y.shape[0] + start
    end = end if end > 0 else y.shape[0] + end

    if not (start < pos <= end):
        raise ValueError(
            f'Provided index {pos} is not between {start=} and {end=}.'
        )

    x_ = X[start:end].copy()
    y_ = y[start:end].copy()
    rel_pos = pos - start

    if feature is not None:
        x_, y_ = sort_by_feature(x_, y_, feature)

    n_samples = y.shape[0]
    n_node_samples = y_.shape[0]

    result = {'pos': pos, 'impurity_parent': impurity(y_)}

    if sample_weight is None:
        result['weighted_n_left'] = float(rel_pos)
        weighted_n_samples = float(n_samples)
        weighted_n_node_samples = float(n_node_samples)
    else:
        result['weighted_n_left'] = sample_weight[start:pos].sum()
        if weighted_n_samples is None:
            weighted_n_samples = sample_weight.sum()
        if weighted_n_node_samples is None:
            weighted_n_node_samples = sample_weight[start:end].sum()

    result['weighted_n_right'] = (
        weighted_n_node_samples - result['weighted_n_left']
    )
    result['impurity_left'] = impurity(y_[:rel_pos])
    result['impurity_right'] = impurity(y_[rel_pos:])

    if average_both_axes:
        result['impurity_parent'] += impurity(y_.T)
        result['impurity_parent'] /= 2
        result['impurity_left'] += impurity(y_[:rel_pos].T)
        result['impurity_left'] /= 2
        result['impurity_right'] += impurity(y_[rel_pos:].T)
        result['impurity_right'] /= 2

    if supervision is not None:
        u_impurity = unsupervised_impurity or impurity

        root_supervised_impurity = result['impurity_parent']
        result['supervised_impurity_parent'] = root_supervised_impurity

        root_unsupervised_impurity = u_impurity(x_)
        result['unsupervised_impurity_parent'] = root_unsupervised_impurity

        result['impurity_parent'] = 1.0

        result['supervised_impurity_left'] = result['impurity_left']
        result['supervised_impurity_right'] = result['impurity_right']
        result['unsupervised_impurity_left'] = u_impurity(x_[:rel_pos])
        result['unsupervised_impurity_right'] = u_impurity(x_[rel_pos:])

        unsupervised_impurity_left = (
            result['unsupervised_impurity_left']
            / root_unsupervised_impurity
        )
        unsupervised_impurity_right = (
            result['unsupervised_impurity_right']
            / root_unsupervised_impurity
        )

        if average_both_axes:
            # unsup impurity of the other axis is 1.0, since it does not change
            unsupervised_impurity_left = unsupervised_impurity_left/2 + 0.5
            unsupervised_impurity_right = unsupervised_impurity_right/2 + 0.5

        result['impurity_left'] = (
            supervision * result['impurity_left'] / root_supervised_impurity
            + (1 - supervision) * unsupervised_impurity_left
        )
        result['impurity_right'] = (
            supervision * result['impurity_right'] / root_supervised_impurity
            + (1 - supervision) * unsupervised_impurity_right
        )

    result['improvement'] = (
        weighted_n_node_samples / weighted_n_samples
        * (
            result['impurity_parent']
            - result['impurity_left']
            * result['weighted_n_left'] / weighted_n_node_samples
            - result['impurity_right']
            * result['weighted_n_right'] / weighted_n_node_samples
        )
    )

    return result


def manually_eval_all_splits(
    x, y,
    sample_weight=None,
    start=0,
    end=0,
    supervision=None,
    indices=None,
    average_both_axes=False,
    impurity=mse_impurity,
    unsupervised_impurity: Callable[[np.ndarray], float] | None = None,
):
    start = start if start >= 0 else y.shape[0] + start
    end = end if end > 0 else y.shape[0] + end

    if sample_weight is None:
        weighted_n_samples = float(y.shape[0])
        weighted_n_node_samples = float(end - start)
    else:
        weighted_n_samples = sample_weight.sum()
        weighted_n_node_samples = sample_weight[start:end].sum()

    if indices is None:
        indices = range(start + 1, end)

    result = defaultdict(list)

    for pos in indices:
        split = evaluate_split(
            x, y,
            pos=pos,
            start=start,
            end=end,
            supervision=supervision,
            average_both_axes=average_both_axes,
            sample_weight=sample_weight,
            weighted_n_samples=weighted_n_samples,
            weighted_n_node_samples=weighted_n_node_samples,
            impurity=impurity,
            unsupervised_impurity=unsupervised_impurity,
        )
        for k, v in split.items():
            result[k].append(v)

    result = {k: np.array(v) for k, v in result.items()}
    result['weighted_n_samples'] = weighted_n_samples
    result['weighted_n_node_samples'] = weighted_n_node_samples

    return result


def semisupervised_criterion_factory(
    x, y,
    supervised_criterion,
    unsupervised_criterion,
    supervision,
    ss_adapter=None,
):
    criterion = make_semisupervised_criterion(
        supervision=supervision,
        supervised_criterion=supervised_criterion,
        unsupervised_criterion=unsupervised_criterion,
        ss_class=ss_adapter,
        n_samples=y.shape[0],
        n_outputs=y.shape[1],
        n_features=x.shape[1],
    )
    criterion.set_X(np.ascontiguousarray(x, dtype='float64'))
    return criterion


# TODO
def ss_gmo_criterion_factory(
    x, y,
    *,
    n_samples,
    n_features,
    n_outputs,
    supervision,
    supervised_criterion,
    unsupervised_criterion,
    bipartite_adapter,
    ss_criterion=None,  # TODO: AxisSSCompositeCriterion
):
    if ss_criterion is None:
        ss_criterion = _infer_ss_adapter_type(
            supervised_criterion,
            unsupervised_criterion,
        )
    ss_criteria = []

    for ax in range(2):
        ss_criteria.append(
            ss_criterion(
                supervised_criterion=supervised_criterion(
                    n_outputs=n_outputs[ax],
                    n_samples=n_samples[ax],
                ),
                unsupervised_criterion=unsupervised_criterion(
                    n_outputs=n_features[ax],
                    n_samples=n_samples[ax],
                ),
                supervision=supervision,
            )
        )
    
    return bipartite_adapter(
        criterion_rows=ss_criteria[0],
        criterion_cols=ss_criteria[1],
    )


def test_no_axis_criterion_as_unsupervised():
    with pytest.raises(TypeError, match=r'.*AxisCriterion'):
        make_semisupervised_criterion(
            supervised_criterion=AxisMSE,
            unsupervised_criterion=AxisMSE,
            n_features=100,
            n_samples=200,
            n_outputs=10,
            supervision=0.5,
        )


@pytest.fixture(
    params=[
        (
            make_2dss_criterion,
            {
                'supervised_criteria': MSE,
                'unsupervised_criteria': MSE,
                'ss_criteria': SSCompositeCriterion,
                'criterion_wrapper_class': SquaredErrorGSO,
            },
            global_mse_impurity,
            mse_impurity,
        ),
        (
            make_2dss_criterion,
            {
                'supervised_criteria': MSE,
                'unsupervised_criteria': MSE,
                'ss_criteria': HomogeneousCompositeSS,
                'criterion_wrapper_class': SquaredErrorGSO,
            },
            global_mse_impurity,
            mse_impurity,
        ),
        # (
        #     make_2dss_criterion,
        #     {
        #         'supervised_criteria': MSE,
        #         'unsupervised_criteria': MSE,
        #         'ss_criteria': HomogeneousCompositeSS,
        #         'criterion_wrapper_class': GMOSA,
        #     },
        #     global_mse_impurity,
        #     mse_impurity,
        # ),
    ],
    ids=['mse', 'homogeneus_mse'],
)
def gso_bipartite_criterion(request, n_samples, n_features, supervision):
    factory, args, supervised_impurity, unsupervised_impurity = request.param

    default_criterion_args = dict(
        n_features=n_features,
        n_samples=n_samples,
        n_outputs=1,
        supervision=supervision,
    )

    return {
        'criterion': factory(**default_criterion_args | args),
        'supervised_impurity': supervised_impurity,
        'unsupervised_impurity': unsupervised_impurity,
    }


def assert_ss_bipartite_criterion_identities(criterion_data):
    criterion = criterion_data['criterion']

    assert (
        criterion.supervised_criterion_rows
        is criterion.ss_criterion_rows.supervised_criterion
    )
    assert (
        criterion.unsupervised_criterion_rows
        is criterion.ss_criterion_rows.unsupervised_criterion
    )
    assert (
        criterion.supervised_criterion_cols
        is criterion.ss_criterion_cols.supervised_criterion
    )
    assert (
        criterion.unsupervised_criterion_cols
        is criterion.ss_criterion_cols.unsupervised_criterion
    )
    assert (
        criterion.supervised_criterion_rows
        is criterion.supervised_bipartite_criterion.criterion_rows
    )
    assert (
        criterion.supervised_criterion_cols
        is criterion.supervised_bipartite_criterion.criterion_cols
    )


def test_criterion_identities_ss_bipartite(gso_bipartite_criterion):
    assert_ss_bipartite_criterion_identities(gso_bipartite_criterion)


@pytest.mark.parametrize(
    'supervised_criterion, unsupervised_criterion, provided_type, target_type',
    [
        (MSE, MSE, None, HomogeneousCompositeSS),
        (AxisMSE, AxisMSE, None, AxisHomogeneousCompositeSS),
        (MSE, FriedmanMSE, None, SSCompositeCriterion),
        (MSE, FriedmanMSE, HomogeneousCompositeSS, HomogeneousCompositeSS),
        (AxisMSE, AxisFriedmanMSE, AxisHomogeneousCompositeSS,
         AxisHomogeneousCompositeSS),
    ],
)
def test_ss_criterion_factory_adapter_type_inference(
    supervised_criterion,
    unsupervised_criterion,
    provided_type,
    target_type,
):
    criterion = make_semisupervised_criterion(
        supervised_criterion=supervised_criterion,
        unsupervised_criterion=unsupervised_criterion,
        ss_class=provided_type,
        n_features=100,
        n_samples=200,
        n_outputs=10,
        supervision=0.5,
    )
    assert type(criterion) == target_type


@pytest.mark.parametrize(
    'criterion_factory, criterion_args, impurity', [
        (
            semisupervised_criterion_factory,
            {
                'supervised_criterion': MSE,
                'unsupervised_criterion': MSE,
                'ss_adapter': SSCompositeCriterion,
            },
            mse_impurity,
        ),
        (
            semisupervised_criterion_factory,
            {'supervised_criterion': MSE, 'unsupervised_criterion': MSE},
            mse_impurity,
        ),
    ],
    ids=['ss', 'ss_homogeneous'],
)
def test_semisupervised_criterion(
    supervision,
    data,
    criterion_factory,
    criterion_args,
    impurity,
    unsupervised_impurity=None,
):
    XX, Y, *_ = data
    X = XX[0]

    # Because x is converted from float32
    rtol = 1e-7 if supervision is None else 1e-4
    atol = 1e-7

    criterion = criterion_factory(
        X, Y, supervision=supervision, **criterion_args,
    )
    split_data = apply_criterion(criterion, Y)

    manual_split_data = manually_eval_all_splits(
        X, Y,
        supervision=supervision,
        impurity=impurity,
        unsupervised_impurity=unsupervised_impurity,
    )

    assert_equal_dicts(
        split_data,
        manual_split_data,
        rtol=rtol,
        atol=atol,
        ignore={
            'n_samples',
            'n_node_samples',
            'proxy_improvement',
            'supervised_impurity_parent',
            'supervised_impurity_left',
            'supervised_impurity_right',
            'unsupervised_impurity_parent',
            'unsupervised_impurity_left',
            'unsupervised_impurity_right',
            'n_outputs',
            'end',
            'start',
        },
        differing_keys='raise',
    )

    # Test that the order of the proxy improvement values matches the order of
    # the final improvement values.
    assert_ranks_are_close(
        split_data['proxy_improvement'],
        manual_split_data['improvement'],
        msg_prefix='(rows proxy) ',
        rtol=rtol,
        atol=atol,
    )


def test_supervised_criterion(data):
    XX, Y, *_ = data
    X = XX[0]

    criterion = MSE(
        n_samples=Y.shape[0],
        n_outputs=Y.shape[1],
    )

    split_data = apply_criterion(criterion, Y)
    manual_split_data = manually_eval_all_splits(X, Y)
    assert_equal_dicts(
        split_data,
        manual_split_data,
        rtol=1e-7,
        atol=1e-8,
        differing_keys='raise',
        ignore={
            'proxy_improvement',
            'n_outputs',
            'n_samples',
            'n_node_samples',
            'start',
            'end',
        },
    )

    # Test that the order of the proxy improvement values matches the order of
    # the final improvement values.
    assert_ranks_are_close(
        split_data['proxy_improvement'],
        manual_split_data['improvement'],
        msg_prefix='(proxy) ',
    )


def test_bipartite_ss_criterion_gso(
    data,
    n_samples,
    gso_bipartite_criterion,
    supervision,
    apply_criterion=apply_bipartite_ss_criterion,
):
    start_row, start_col = 0, 0
    end_row, end_col = n_rows, n_cols = n_samples

    # Because x is converted from float32
    rtol = 1e-7 if supervision is None else 1e-4
    atol = 1e-7

    X, Y, *_ = data

    criterion = gso_bipartite_criterion['criterion']
    impurity = gso_bipartite_criterion['supervised_impurity']
    unsupervised_impurity = gso_bipartite_criterion['unsupervised_impurity']

    row_splits, col_splits = apply_criterion(
        criterion,
        X_rows=X[0],
        X_cols=X[1],
        y=Y,
        start=[start_row, start_col],
        end=[end_row, end_col],
    )

    # Split positions to evaluate
    row_indices = np.arange(start_row + 1, end_row)
    col_indices = np.arange(start_col + 1, end_col)

    row_manual_splits = manually_eval_all_splits(
        x=np.repeat(X[0], n_cols, axis=0),
        y=Y.reshape(-1, 1),
        supervision=supervision,
        indices=row_indices * n_cols,
        start=start_row * n_cols,
        end=end_row * n_cols,
        impurity=impurity,
        unsupervised_impurity=unsupervised_impurity,
        average_both_axes=True,
    )
    col_manual_splits = manually_eval_all_splits(
        x=np.repeat(X[1], n_rows, axis=0),
        y=Y.T.reshape(-1, 1),
        supervision=supervision,
        indices=col_indices * n_rows,
        start=start_col * n_rows,
        end=end_col * n_rows,
        impurity=impurity,
        unsupervised_impurity=unsupervised_impurity,
        average_both_axes=True,
    )
    row_manual_splits['pos'] = row_indices
    col_manual_splits['pos'] = col_indices

    if supervision is not None:
        row_manual_splits['supervised_pos'] = row_indices
        row_manual_splits['unsupervised_pos'] = row_indices
        col_manual_splits['supervised_pos'] = col_indices
        col_manual_splits['unsupervised_pos'] = col_indices

    ignore = {
        'weighted_n_node_samples_axis',
        'weighted_n_samples_axis',
        'weighted_n_left',
        'weighted_n_right',
    }

    # Test that the order of the proxy improvement values matches the order of
    # the final improvement values.
    assert_ranks_are_close(
        row_splits['proxy_improvement'],
        row_manual_splits['improvement'],
        msg_prefix='(rows proxy) ',
        rtol=rtol,
        atol=atol,
    )
    assert_ranks_are_close(
        col_splits['proxy_improvement'],
        col_manual_splits['improvement'],
        msg_prefix='(cols proxy) ',
        rtol=rtol,
        atol=atol,
    )

    assert_equal_dicts(
        row_splits,
        row_manual_splits,
        msg_prefix='(rows) ',
        subset=row_manual_splits.keys(),
        ignore=ignore,
        rtol=rtol,
        atol=atol,
        differing_keys='raise',
    )
    assert_equal_dicts(
        col_splits,
        col_manual_splits,
        msg_prefix='(cols) ',
        subset=row_manual_splits.keys(),
        ignore=ignore,
        rtol=rtol,
        atol=atol,
        differing_keys='raise',
    )


@pytest.mark.parametrize(
    'criterion_factory, criterion_args, impurity', [
        (
            make_bipartite_criterion,
            {
                'criteria': MSE,
                'criterion_wrapper_class': SquaredErrorGSO,
            },
            global_mse_impurity,
        ),
        (
            make_bipartite_criterion,
            {
                'criteria': AxisSquaredErrorGSO,
                'criterion_wrapper_class': GMOSA,
            },
            global_mse_impurity,
        ),
    ],
    ids=['mse', 'gmo_mse'],
)
def test_bipartite_criterion_gso(
    data,
    n_samples,
    criterion_factory,
    criterion_args,
    impurity,
    default_criterion_args=None,
):
    default_criterion_args = default_criterion_args or dict(
        n_samples=n_samples,
        n_outputs=(
            n_samples[::-1] if criterion_args.get('criterion_wrapper_class')
            in (GMOSA, GMO) else 1
        ),
    )
    criterion = criterion_factory(**default_criterion_args | criterion_args)

    test_bipartite_ss_criterion_gso(
        data=data,
        n_samples=n_samples,
        supervision=None,
        gso_bipartite_criterion={
            'criterion': criterion,
            'supervised_impurity': impurity,
            'unsupervised_impurity': None,
        },
        apply_criterion=apply_bipartite_criterion,
    )


@pytest.mark.parametrize(
    'criterion_factory, criterion_args, impurity, unsupervised_impurity', [
        (
            make_2dss_criterion,
            {
                'supervised_criteria': AxisMSE,
                'unsupervised_criteria': MSE,
                'ss_criteria': SSCompositeCriterion,
                'criterion_wrapper_class': GMOSA,
            },
            mse_impurity,
            None,
        ),
    ],
    ids=['mse'],
)
def test_bipartite_ss_criterion_gmo(
    data,
    n_features,
    n_samples,
    criterion_factory,
    criterion_args,
    impurity,
    unsupervised_impurity,
    supervision,
):
    start_row, start_col = 0, 0
    end_row, end_col = n_samples

    # Because x is converted from float32
    rtol = 1e-7 if supervision is None else 1e-4
    atol = 1e-7

    X, Y, *_ = data

    default_args = dict(
        n_features=n_features,
        n_samples=n_samples,
        n_outputs=sum(n_samples),
        supervision=supervision,
    )
    criterion = criterion_factory(**default_args | criterion_args)

    row_splits, col_splits = apply_bipartite_ss_criterion(
        criterion,
        X[0],
        X[1],
        Y,
        start=[start_row, start_col],
        end=[end_row, end_col],
    )

    row_manual_splits = manually_eval_all_splits(
        x=X[0], y=Y,
        supervision=supervision,
        start=start_row,
        end=end_row,
        average_both_axes=True,
        impurity=impurity,
        unsupervised_impurity=unsupervised_impurity,
    )
    col_manual_splits = manually_eval_all_splits(
        x=X[1], y=Y.T,
        supervision=supervision,
        start=start_col,
        end=end_col,
        average_both_axes=True,
        impurity=impurity,
        unsupervised_impurity=unsupervised_impurity,
    )

    row_manual_splits['axis_weighted_n_samples'] = \
        row_manual_splits['weighted_n_samples']
    col_manual_splits['axis_weighted_n_samples'] = \
        col_manual_splits['weighted_n_samples']
    row_manual_splits['axis_weighted_n_node_samples'] = \
        row_manual_splits['weighted_n_node_samples']
    col_manual_splits['axis_weighted_n_node_samples'] = \
        col_manual_splits['weighted_n_node_samples']

    ignore = {'weighted_n_samples', 'weighted_n_node_samples'}

    assert_equal_dicts(
        row_splits,
        row_manual_splits,
        msg_prefix='(rows) ',
        subset=row_manual_splits.keys(),
        ignore=ignore,
        rtol=rtol,
        atol=atol,
        differing_keys='raise',
    )
    assert_equal_dicts(
        col_splits,
        col_manual_splits,
        msg_prefix='(cols) ',
        subset=row_manual_splits.keys(),
        ignore=ignore,
        rtol=rtol,
        atol=atol,
        differing_keys='raise',
    )

    # Test that the order of the proxy improvement values matches the order of
    # the final improvement values.
    assert_ranks_are_close(
        row_splits['proxy_improvement'],
        row_manual_splits['improvement'],
        msg_prefix='(rows proxy) ',
        rtol=rtol,
        atol=atol,
    )
    assert_ranks_are_close(
        col_splits['proxy_improvement'],
        col_manual_splits['improvement'],
        msg_prefix='(cols proxy) ',
        rtol=rtol,
        atol=atol,
    )
