# TODO: test Friedman Criterion
from typing import Callable, Dict
import numpy as np
import pytest
import scipy.stats

from collections import defaultdict
from numbers import Real, Integral
from sklearn.utils.validation import check_random_state
from sklearn.utils._param_validation import validate_params, Interval
from hypertrees.tree._splitter_factory import (
    make_semisupervised_criterion,
    make_bipartite_criterion,
    make_2dss_criterion,
)
from hypertrees.tree._axis_criterion import (
    AxisSquaredError,
    AxisSquaredErrorGSO,
    AxisFriedmanGSO,
    AxisGini,
    AxisEntropy,
)
from hypertrees.tree._unsupervised_criterion import (
    PairwiseFriedman,
    PairwiseSquaredError,
    PairwiseSquaredErrorGSO,
    UnsupervisedSquaredError,
    UnsupervisedGini,
    UnsupervisedEntropy,
    # UnsupervisedSquaredErrorGSO,  # TODO
    UnsupervisedFriedman,
)
from hypertrees.tree._nd_criterion import GMO, GMOSA
from hypertrees.tree._semisupervised_criterion import (
    SSCompositeCriterion,
)

from hypertrees.melter import row_cartesian_product
from test_utils import assert_equal_dicts, comparison_text
from make_examples import make_interaction_blobs
from splitter_test import (
    apply_criterion,
    apply_ss_criterion,
    apply_bipartite_criterion,
    apply_bipartite_ss_criterion,
)

CLASSIFICATION_CRITERIA = {
    AxisGini,
    AxisEntropy,
    UnsupervisedGini,
    UnsupervisedEntropy,
}


# =============================================================================
# General fixtures
# =============================================================================


@pytest.fixture(params=[0.0, 0.00001, 0.1, 0.2328, 0.569, 0.782, 0.995, 1.0])
def supervision(request):
    return request.param


@pytest.fixture(params=range(10))
def random_state(request):
    return check_random_state(request.param)


@pytest.fixture
def n_samples(request):
    return (11, 7)


@pytest.fixture
def n_classes(n_samples):
    return (
        np.repeat(6, n_samples[1]),
        np.repeat(6, n_samples[0]),
    )


@pytest.fixture(params=[(3, 5), (2, 3)])
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
    X = [Xi.astype('float64') for Xi in X]
    return X, Y, x, y


# =============================================================================
# Manually defined impurity functions
# =============================================================================


def gini_impurity(y):
    sq_sums = 0
    for col in y.T:
        _, counts = np.unique(col, return_counts=True)
        sq_sums += np.square(counts / y.shape[0]).sum()
    return 1 - sq_sums / y.shape[1]


def entropy_impurity(y):
    col_entropies = 0
    for col in y.T:
        _, counts = np.unique(col, return_counts=True)
        p = counts / y.shape[0]
        col_entropies -= (p * np.log2(p)).sum()
    return col_entropies / y.shape[1]


def mse_impurity(y):
    return y.var(0).mean()


def global_mse_impurity(y):
    return y.var()


# =============================================================================
# Criterion fixtures and corresponding impurity functions
# =============================================================================


@pytest.fixture(
    params=[
        (
            make_semisupervised_criterion,
            {
                'supervised_criterion': UnsupervisedGini,
                'unsupervised_criterion': UnsupervisedSquaredError,
                'ss_class': SSCompositeCriterion,
            },
            gini_impurity,
            mse_impurity,
        ),
        (
            make_semisupervised_criterion,
            {
                'supervised_criterion': UnsupervisedEntropy,
                'unsupervised_criterion': UnsupervisedSquaredError,
                'ss_class': SSCompositeCriterion,
            },
            entropy_impurity,
            mse_impurity,
        ),
        (
            make_semisupervised_criterion,
            {
                'supervised_criterion': UnsupervisedSquaredError,
                'unsupervised_criterion': UnsupervisedSquaredError,
            },
            mse_impurity,
            mse_impurity,
        ),
    ],
    ids=['gini_mse', 'entropy_mse', 'mse'],
)
def semisupervised_criterion(
    request,
    n_samples,
    n_features,
    n_classes,
    supervision,
):
    factory, args, supervised_impurity, unsupervised_impurity = request.param

    default_criterion_args = dict(
        n_features=n_features[0],
        n_samples=n_samples[0],
        n_classes=n_classes[0],
        n_outputs=n_samples[1],
        supervision=supervision,
    )

    return {
        'criterion': factory(**default_criterion_args | args),
        'supervised_impurity': supervised_impurity,
        'unsupervised_impurity': unsupervised_impurity,
    }


@pytest.fixture(
    params=[
        (
            make_2dss_criterion,
            {
                'supervised_criteria': AxisGini,
                'unsupervised_criteria': UnsupervisedSquaredError,
                'ss_criteria': SSCompositeCriterion,
                'criterion_wrapper_class': GMOSA,
            },
            gini_impurity,
            mse_impurity,
        ),
        (
            make_2dss_criterion,
            {
                'supervised_criteria': AxisEntropy,
                'unsupervised_criteria': UnsupervisedSquaredError,
                'ss_criteria': SSCompositeCriterion,
                'criterion_wrapper_class': GMOSA,
            },
            entropy_impurity,
            mse_impurity,
        ),
        (
            make_2dss_criterion,
            {
                'supervised_criteria': AxisSquaredErrorGSO,
                'unsupervised_criteria': UnsupervisedSquaredError,
                'ss_criteria': SSCompositeCriterion,
                'criterion_wrapper_class': GMOSA,
            },
            global_mse_impurity,
            mse_impurity,
        ),
        (
            make_2dss_criterion,
            {
                'supervised_criteria': AxisSquaredError,
                'unsupervised_criteria': UnsupervisedSquaredError,
                'ss_criteria': SSCompositeCriterion,
                'criterion_wrapper_class': GMOSA,
            },
            mse_impurity,
            mse_impurity,
        ),
    ],
    ids=['gini_mse', 'entropy_mse', 'mse_gso', 'mse_gmo'],
)
def ss_bipartite_criterion(
    request,
    n_samples,
    n_features,
    supervision,
    n_classes,
):
    factory, args, supervised_impurity, unsupervised_impurity = request.param

    default_criterion_args = dict(
        n_features=n_features,
        n_samples=n_samples,
        n_classes=n_classes,
        n_outputs=n_samples[::-1],
        supervision=supervision,
    )

    return {
        'criterion': factory(**default_criterion_args | args),
        'supervised_impurity': supervised_impurity,
        'unsupervised_impurity': unsupervised_impurity,
    }


@pytest.fixture(
    params=[
        (
            make_bipartite_criterion,
            {
                'criteria': AxisGini,
                'criterion_wrapper_class': GMOSA,
            },
            gini_impurity,
        ),
        (
            make_bipartite_criterion,
            {
                'criteria': AxisEntropy,
                'criterion_wrapper_class': GMOSA,
            },
            entropy_impurity,
        ),
        (
            make_bipartite_criterion,
            {
                'criteria': AxisSquaredErrorGSO,
                'criterion_wrapper_class': GMOSA,
            },
            global_mse_impurity,
        ),
        (
            make_bipartite_criterion,
            {
                'criteria': AxisSquaredError,
                'criterion_wrapper_class': GMOSA,
            },
            mse_impurity,
        ),
    ],
    ids=['gini', 'entropy', 'mse_gso', 'mse_gmo'],
)
def supervised_bipartite_criterion(request, n_samples, n_classes):
    factory, args, impurity = request.param

    default_criterion_args = dict(
        n_samples=n_samples,
        n_outputs=n_samples[::-1],
        n_classes=n_classes,
    )
    return {
        'criterion': factory(**default_criterion_args | args),
        'supervised_impurity': impurity,
        'unsupervised_impurity': None,
    }


@pytest.fixture
def semisupervised_splits(
    semisupervised_criterion,
    data,
    supervision,
):
    return get_ss_splits(
        ss_criterion_data=semisupervised_criterion,
        data=data,
        supervision=supervision,
        apply_criterion=apply_ss_criterion,
    )


@pytest.fixture
def supervised_splits(
    criterion,
    data,
    n_samples,
):
    return get_ss_splits(
        ss_criterion_data=criterion,
        data=data,
        supervision=None,
        apply_criterion=apply_criterion,
    )


@pytest.fixture
def ss_bipartite_splits(
    ss_bipartite_criterion,
    data,
    n_samples,
    supervision,
):
    return get_ss_bipartite_splits(
        ss_bipartite_criterion=ss_bipartite_criterion,
        data=data,
        n_samples=n_samples,
        supervision=supervision,
        apply_criterion=apply_bipartite_ss_criterion,
    )


@pytest.fixture
def supervised_bipartite_splits(
    supervised_bipartite_criterion,
    n_samples,
    data,
):
    return get_ss_bipartite_splits(
        ss_bipartite_criterion=supervised_bipartite_criterion,
        data=data,
        n_samples=n_samples,
        supervision=None,
        apply_criterion=apply_bipartite_criterion,
    )


# =============================================================================
# Utility functions
# =============================================================================


def turn_into_classification(y):
    n_bins = 5
    y = np.digitize(y, np.linspace(y.min(), y.max(), n_bins))
    return y.astype('float64')


def get_ss_splits(
    *,
    ss_criterion_data,
    data,
    supervision,
    apply_criterion,
):
    XX, Y, *_ = data
    X = XX[0].astype('float64')

    criterion = ss_criterion_data['criterion']
    supervised_impurity = ss_criterion_data['supervised_impurity']
    unsupervised_impurity = ss_criterion_data['unsupervised_impurity']

    if (
        type(criterion) in CLASSIFICATION_CRITERIA
        or supervision is not None
        and type(criterion.supervised_criterion) in CLASSIFICATION_CRITERIA
    ):
        Y = turn_into_classification(Y)

    criterion.set_X(X)

    split_data = apply_criterion(criterion, Y)

    manual_split_data = manually_eval_all_splits(
        X, Y,
        supervision=supervision,
        impurity=supervised_impurity,
        unsupervised_impurity=unsupervised_impurity,
    )

    return {'splits': split_data, 'reference_splits': manual_split_data}


def get_ss_bipartite_splits(
    *,
    ss_bipartite_criterion,
    data,
    n_samples,
    supervision,
    apply_criterion,
):
    X, Y, *_ = data

    start_row, start_col = 0, 0
    end_row, end_col = n_samples

    criterion = ss_bipartite_criterion['criterion']
    supervised_impurity = ss_bipartite_criterion['supervised_impurity']
    unsupervised_impurity = ss_bipartite_criterion['unsupervised_impurity']

    if (
        type(criterion.criterion_rows) in CLASSIFICATION_CRITERIA
        or supervision is not None
        and type(criterion.criterion_rows.supervised_criterion)
        in CLASSIFICATION_CRITERIA
    ):
        Y = turn_into_classification(Y)

    row_splits, col_splits = apply_criterion(
        criterion,
        X[0],
        X[1],
        Y,
        start=[start_row, start_col],
        end=[end_row, end_col],
    )

    ref_row_splits = manually_eval_all_splits(
        x=X[0], y=Y,
        supervision=supervision,
        start=start_row,
        end=end_row,
        average_both_axes=True,
        impurity=supervised_impurity,
        unsupervised_impurity=unsupervised_impurity,
    )
    ref_col_splits = manually_eval_all_splits(
        x=X[1], y=Y.T,
        supervision=supervision,
        start=start_col,
        end=end_col,
        average_both_axes=True,
        impurity=supervised_impurity,
        unsupervised_impurity=unsupervised_impurity,
    )

    return {
        'splits': (row_splits, col_splits),
        'reference_splits': (ref_row_splits, ref_col_splits),
    }


def assert_correct_proxy_factors(ss_splits, **kwargs):
    sup_data = scipy.stats.linregress(
        ss_splits['supervised_improvement'],
        ss_splits['supervised_proxy_improvement'],
    )
    unsup_data = scipy.stats.linregress(
        ss_splits['unsupervised_improvement'],
        ss_splits['unsupervised_proxy_improvement'],
    )

    assert sup_data.pvalue < 1e-3, "Uncorrelated supervised proxies."
    assert unsup_data.pvalue < 1e-3, "Uncorrelated unsupervised proxies."
    assert np.unique(ss_splits['supervised_proxy_factor']).shape[0] == 1, (
        "Supervised proxy factor is not constant!"
    )
    assert np.unique(ss_splits['unsupervised_proxy_factor']).shape[0] == 1, (
        "Unsupervised proxy factor is not constant!"
    )

    assert_equal_dicts(
        {
            'sup_proxy_factor': ss_splits['supervised_proxy_factor'][0],
            'unsup_proxy_factor': ss_splits['unsupervised_proxy_factor'][0],
        },
        {
            'sup_proxy_factor': sup_data.slope,
            'unsup_proxy_factor': unsup_data.slope,
        },
        **kwargs,
    )


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


# =============================================================================
# Tests
# =============================================================================


def test_no_axis_criterion_as_unsupervised():
    with pytest.raises(TypeError, match=r'.*AxisCriterion'):
        make_semisupervised_criterion(
            supervised_criterion=AxisSquaredError,
            unsupervised_criterion=AxisSquaredError,
            n_features=100,
            n_samples=200,
            n_outputs=10,
            supervision=0.5,
        )


def test_ss_criterion(semisupervised_splits):
    rtol = 1e-7
    atol = 1e-7

    assert_equal_dicts(
        semisupervised_splits['splits'],
        semisupervised_splits['reference_splits'],
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
        subset=semisupervised_splits.keys(),
        differing_keys='raise',
    )


def test_ss_criterion_proxies(semisupervised_splits):
    """Test semi-supervised proxy_impurities.

    Test that the order of the proxy improvement values matches the order of
    the final improvement values.
    """
    rtol = 1e-7
    atol = 1e-7

    splits = semisupervised_splits['splits']
    ref_splits = semisupervised_splits['reference_splits']

    assert_ranks_are_close(
        splits['supervised_proxy_improvement'],
        splits['supervised_improvement'],
        msg_prefix='(supervised proxy) ',
        rtol=rtol,
        atol=atol,
    )
    assert_ranks_are_close(
        splits['unsupervised_proxy_improvement'],
        splits['unsupervised_improvement'],
        msg_prefix='(unsupervised proxy) ',
        rtol=rtol,
        atol=atol,
    )
    assert_ranks_are_close(
        splits['ss_proxy_improvement'],
        splits['ss_improvement'],
        msg_prefix='(proxy vs. improvement) ',
        rtol=rtol,
        atol=atol,
    )
    assert_ranks_are_close(
        splits['ss_proxy_improvement'],
        ref_splits['improvement'],
        msg_prefix='(proxy vs. reference) ',
        rtol=rtol,
        atol=atol,
    )


def test_supervised_criterion(data):
    XX, Y, *_ = data
    X = XX[0]

    criterion = UnsupervisedSquaredError(
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


def test_ss_bipartite_criterion(ss_bipartite_splits):
    rtol = 1e-7
    atol = 1e-7

    row_splits, col_splits = ss_bipartite_splits['splits']
    ref_row_splits, ref_col_splits = ss_bipartite_splits['reference_splits']

    ref_row_splits['axis_weighted_n_samples'] = \
        ref_row_splits['weighted_n_samples']
    ref_col_splits['axis_weighted_n_samples'] = \
        ref_col_splits['weighted_n_samples']
    ref_row_splits['axis_weighted_n_node_samples'] = \
        ref_row_splits['weighted_n_node_samples']
    ref_col_splits['axis_weighted_n_node_samples'] = \
        ref_col_splits['weighted_n_node_samples']

    ignore = {'weighted_n_samples', 'weighted_n_node_samples'}

    assert_equal_dicts(
        row_splits,
        ref_row_splits,
        msg_prefix='(rows) ',
        subset=ref_row_splits.keys(),
        ignore=ignore,
        rtol=rtol,
        atol=atol,
        differing_keys='raise',
    )
    assert_equal_dicts(
        col_splits,
        ref_col_splits,
        msg_prefix='(cols) ',
        subset=ref_row_splits.keys(),
        ignore=ignore,
        rtol=rtol,
        atol=atol,
        differing_keys='raise',
    )


def test_ss_bipartite_proxy_improvement(ss_bipartite_splits):
    """Test bipartite semi-supervised proxy_impurities.

    Test that the order of the proxy improvement values matches the order of
    the final improvement values.
    """
    # Test that the order of the proxy improvement values matches the order of
    # the final improvement values.
    rtol = 1e-7
    atol = 1e-7

    for axis, (splits, ref_splits) in enumerate(zip(
        ss_bipartite_splits['splits'],
        ss_bipartite_splits['reference_splits'],
    )):
        # Perform control test asserting that the impurity improvement matches
        # the manually calculated reference values.
        assert_ranks_are_close(
            splits['improvement'],
            ref_splits['improvement'],
            msg_prefix=f'({axis=}, bipartite vs reference) ',
            rtol=rtol,
            atol=atol,
        )

        # Additional tests for better inspection of semisupervised splits.
        if 'ss_improvement' in splits:
            assert_ranks_are_close(
                splits['ss_improvement'],
                splits['improvement'],
                msg_prefix=f'({axis=}, axis vs bipartite) ',
                rtol=rtol,
                atol=atol,
            )
            assert_ranks_are_close(
                splits['supervised_proxy_improvement'],
                splits['supervised_improvement'],
                msg_prefix=f'({axis=}, sup proxy vs axis) ',
                rtol=rtol,
                atol=atol,
            )
            assert_ranks_are_close(
                splits['unsupervised_proxy_improvement'],
                splits['unsupervised_improvement'],
                msg_prefix=f'({axis=}, unsup proxy vs axis) ',
                rtol=rtol,
                atol=atol,
            )
            assert_ranks_are_close(
                splits['ss_proxy_improvement'],
                splits['ss_improvement'],
                msg_prefix=f'({axis=}, proxy vs axis) ',
                rtol=rtol,
                atol=atol,
            )

        # Final main assertion.
        assert_ranks_are_close(
            splits['proxy_improvement'],
            ref_splits['improvement'],
            msg_prefix=f'({axis=}, proxy vs reference) ',
            rtol=rtol,
            atol=atol,
        )


def test_supervised_bipartite_criterion(supervised_bipartite_splits):
    test_ss_bipartite_criterion(supervised_bipartite_splits)


def test_supervised_bipartite_proxy_improvement(supervised_bipartite_splits):
    test_ss_bipartite_proxy_improvement(supervised_bipartite_splits)


def test_monopartite_proxy_factors(semisupervised_splits):
    assert_correct_proxy_factors(semisupervised_splits['splits'])


def test_bipartite_proxy_factors(ss_bipartite_splits):
    row_splits, col_splits = ss_bipartite_splits['splits']
    assert_correct_proxy_factors(row_splits, msg_prefix='(rows) ')
    assert_correct_proxy_factors(col_splits, msg_prefix='(cols) ')


@pytest.mark.parametrize(
    'single_output_impurity, multioutput_impurity', [
        (global_mse_impurity, mse_impurity),
    ],
    ids=['mse'],
)
def test_gso_gmo_equivalence(
    data,
    n_samples,
    supervision,
    single_output_impurity,
    multioutput_impurity,
):
    """Tests if a single-output impurity metric is invariant in the GSO format.

    The test ensures that a single-output version of a metric (that ignores
    different columns as different outputs) yields the same values on the two
    formats of a bipartite dataset described bellow.

        1. Global Multi-Output (GMO) format: 
            Axis 0 (rows) receives:
                X = X[0]
                Y = Y
            Axis 1 (columns) receives:
                X = X[1]
                Y = Y.T

        2. Global Single Output (GSO) format: 
            Axis 0 (rows) receives:
                X = np.repeat(X[0], n_cols, axis=0)
                Y = Y.reshape(-1, 1))
            Axis 1 (columns) receives:
                X = np.repeat(X[1], n_rows, axis=0)
                Y = Y.reshape(-1, 1))

    An example of single-output impurity is the UnsupervisedSquaredError defined as Y.var() instead
    of its more usual multioutput form Y.var(0).mean().

    If the metric passes this test, we can employ it in testing bipartite
    criteria directly in the more convenient GMO format (done by
    `test_semisupervised_bipartite_criterion()`) being sure the criterion being compared to
    the metric would yield the same results if the multi-output counterpart of
    the metric were to be applied in the GSO-formatted dataset.
    """
    start_row, start_col = 0, 0
    end_row, end_col = n_samples
    n_rows, n_cols = n_samples

    rtol = 1e-7
    atol = 1e-7

    X, Y, *_ = data

    # Split positions to evaluate
    row_indices = np.arange(start_row + 1, end_row)
    col_indices = np.arange(start_col + 1, end_col)

    so_row_splits = manually_eval_all_splits(
        x=np.repeat(X[0], n_cols, axis=0),
        y=Y.reshape(-1, 1),
        supervision=supervision,
        indices=row_indices * n_cols,
        start=start_row * n_cols,
        end=end_row * n_cols,
        impurity=multioutput_impurity,
        unsupervised_impurity=multioutput_impurity,
    )
    so_col_splits = manually_eval_all_splits(
        x=np.repeat(X[1], n_rows, axis=0),
        y=Y.T.reshape(-1, 1),
        supervision=supervision,
        indices=col_indices * n_rows,
        start=start_col * n_rows,
        end=end_col * n_rows,
        impurity=multioutput_impurity,
        unsupervised_impurity=multioutput_impurity,
    )

    mo_row_splits = manually_eval_all_splits(
        x=X[0], y=Y,
        supervision=supervision,
        start=start_row,
        end=end_row,
        impurity=single_output_impurity,
        unsupervised_impurity=multioutput_impurity,
    )
    mo_col_splits = manually_eval_all_splits(
        x=X[1], y=Y.T,
        supervision=supervision,
        start=start_col,
        end=end_col,
        impurity=single_output_impurity,
        unsupervised_impurity=multioutput_impurity,
    )

    so_row_splits['pos'] = row_indices
    so_col_splits['pos'] = col_indices

    ignore = {
        'weighted_n_left',
        'weighted_n_right',
        'weighted_n_samples',
        'weighted_n_node_samples',
    }
    assert_equal_dicts(
        so_row_splits,
        mo_row_splits,
        msg_prefix='(rows) ',
        ignore=ignore,
        rtol=rtol,
        atol=atol,
        differing_keys='raise',
    )
    assert_equal_dicts(
        so_col_splits,
        mo_col_splits,
        msg_prefix='(cols) ',
        ignore=ignore,
        rtol=rtol,
        atol=atol,
        differing_keys='raise',
    )
