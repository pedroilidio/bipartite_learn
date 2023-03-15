import logging
import numpy as np
import pytest
from numbers import Real, Integral
from sklearn.utils.validation import check_random_state
from sklearn.utils._testing import assert_allclose
from sklearn.utils._param_validation import validate_params, Interval
from sklearn.tree._criterion import MSE, MAE, FriedmanMSE, Poisson
from sklearn.tree._splitter import BestSplitter
from hypertrees.tree._splitter_factory import (
    make_semisupervised_criterion,
    make_2dss_criterion,
)
from hypertrees.tree._nd_criterion import SquaredErrorGSO
from hypertrees.melter import row_cartesian_product
from test_utils import assert_equal_dicts, comparison_text
from make_examples import make_interaction_blobs
from splitter_test import (
    apply_criterion,
    apply_bipartite_ss_criterion,
    get_criterion_status,
    update_criterion,
)


@pytest.fixture(params=[0.0, 0.00001, 0.1, 0.2328, 0.569, 0.782, 0.995, 1.0])
def supervision(request):
    return request.param


@pytest.fixture(params=range(10))
def random_state(request):
    return request.param


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
        random_state=check_random_state(random_state),
        noise=2.0,
        centers=10,
    )
    return X, Y, x, y


def mse_impurity(a, axis=0):
    """Calculate impurity the same way as MSE criterion.

    Mathematically the same as a.var(axis).mean(), but prone to precision
    issues.
    """
    a_sum = a.sum(axis)
    n_out = a_sum.shape[0]
    n_samples = a.shape[axis]
    return ((a ** 2).sum()/n_samples - (a_sum ** 2).sum()/n_samples**2) / n_out


def global_mse_impurity(a, axis=0):
    """Calculate impurity the same way as MSE criterion.

    Mathematically the same as a.var(axis).mean(), but prone to precision
    issues.
    """
    a_sum = a.sum(axis)
    n_out = a_sum.shape[0]
    n_samples = a.shape[axis]
    return ((a ** 2).sum()/n_samples - a_sum.sum()**2/n_samples**2) / n_out


def sort_by_feature(x, y, feature):
    sorted_indices = x[:, feature].argsort()
    return x[sorted_indices], y[sorted_indices]


@validate_params({
    'X': ['array-like'],
    'y': ['array-like'],
    'start': [Integral],
    'end': [Integral],
    'feature': [Interval(Integral, 0, None, closed='left'), None],
    'supervision': [Interval(Real, 0.0, 1.0, closed='both')],
    'average_both_axes': ['boolean'],
    'impurity': [callable],
})
def manual_split_eval_mse(
    X,
    y,
    pos,  # absolute position, disregarding start and end
    start=0,  # Crops the data between positions
    end=0,
    feature=None,
    supervision=1.0,
    average_both_axes=False,
    impurity=mse_impurity,
):
    start = start if start >= 0 else y.shape[0] + start
    end = end if end > 0 else y.shape[0] + end

    if not (start < pos < end):
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

    result = {'pos': pos, 'impurity_parent': impurity(y_, 0)}

    result['impurity_left'] = impurity(y_[:rel_pos], 0)
    result['impurity_right'] = impurity(y_[rel_pos:], 0)

    if average_both_axes:
        result['impurity_parent'] += impurity(y_, 1)
        result['impurity_parent'] /= 2
        result['impurity_left'] += impurity(y_[:rel_pos], 1)
        result['impurity_left'] /= 2
        result['impurity_right'] += impurity(y_[rel_pos:], 1)
        result['impurity_right'] /= 2

    if supervision < 1.0:
        result['impurity_parent'] *= supervision
        result['impurity_parent'] += (1-supervision) * impurity(x_, 0)
        result['impurity_left'] *= supervision
        result['impurity_left'] += (1-supervision) * impurity(x_[:rel_pos], 0)
        result['impurity_right'] *= supervision
        result['impurity_right'] += (1-supervision) * \
            impurity(x_[rel_pos:], 0)

    n_left = rel_pos
    n_right = n_node_samples - rel_pos

    result['improvement'] = n_node_samples / n_samples * (
        result['impurity_parent']
        - (
            n_left * result['impurity_left']
            + n_right * result['impurity_right']
        ) / n_node_samples
    )

    return result


def manually_eval_all_splits(
    x, y,
    start=0,  # Crops data to select partition, not related to indices
    end=0,
    supervision=1.0,
    indices=None,
    average_both_axes=False,
    impurity=mse_impurity,
):
    result = dict(
        pos=[],
        impurity_parent=[],
        impurity_left=[],
        impurity_right=[],
        improvement=[],
    )
    start = start if start >= 0 else y.shape[0] + start
    end = end if end > 0 else y.shape[0] + end

    if indices is None:
        indices = range(start+1, end)

    for pos in indices:
        split = manual_split_eval_mse(
            x, y,
            pos=pos,
            start=start,
            end=end,
            supervision=supervision,
            average_both_axes=average_both_axes,
            impurity=impurity,
        )
        for k, v in split.items():
            result[k].append(v)

    return {k: np.asarray(v) for k, v in result.items()}


def test_semisupervised_criterion(supervision, data):
    X, Y, x, y = data

    criterion = make_semisupervised_criterion(
        supervision=supervision,
        supervised_criterion=MSE,
        unsupervised_criterion=MSE,
        n_features=x.shape[1],
        n_samples=x.shape[0],
        n_outputs=y.shape[1],
    )

    split_data = apply_criterion(criterion, np.hstack((x, y)))
    manual_split_data = manually_eval_all_splits(x, y, supervision=supervision)
    # rtol=1e-4 because x is converted from float32
    assert_equal_dicts(split_data, manual_split_data, rtol=1e-3, atol=1e-8)


def test_supervised_criterion(data):
    X, Y, x, y = data

    criterion = MSE(
        n_samples=y.shape[0],
        n_outputs=y.shape[1],
    )

    split_data = apply_criterion(criterion, y)
    manual_split_data = manually_eval_all_splits(x, y)
    assert_equal_dicts(split_data, manual_split_data, rtol=1e-7, atol=1e-8)


def test_unsupervised_criterion(data):
    X, Y, x, y = data

    criterion = make_semisupervised_criterion(
        supervision=0.0,
        supervised_criterion=MSE,
        unsupervised_criterion=MSE,
        n_features=x.shape[1],
        n_samples=x.shape[0],
        n_outputs=y.shape[1],
    )

    split_data = apply_criterion(criterion, np.hstack((x, y)))
    manual_split_data = manually_eval_all_splits(x=x, y=x, supervision=1.0)
    # rtol=1e-4 because x is converted from float32
    assert_equal_dicts(split_data, manual_split_data, rtol=1e-3, atol=1e-8)


def test_fake_unsupervised_criterion(data):
    X, Y, x, y = data

    criterion = MSE(
        n_samples=x.shape[0],
        n_outputs=x.shape[1],
    )

    split_data = apply_criterion(criterion, x.astype('float64'))
    manual_split_data = manually_eval_all_splits(x=x, y=x, supervision=1.0)
    # rtol=1e-4 because x is converted from float32
    assert_equal_dicts(split_data, manual_split_data, rtol=1e-3, atol=1e-8)


def test_bipartite_ss_criterion(data, n_features, n_samples, supervision, random_state):
    start_row, start_col = 0, 0
    end_row, end_col = n_samples
    n_rows, n_cols = n_samples

    X, Y, x, y = data
    xT, yT = row_cartesian_product([X[1], X[0]]), Y.T.reshape(-1, 1)

    criterion = make_2dss_criterion(
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        criterion_wrapper_class=SquaredErrorGSO,
        supervision=supervision,
        n_features=n_features,
        n_samples=n_samples,
        n_outputs=1,
        random_state=check_random_state(random_state),
        axis_decision_only=False,
    )

    row_splits, col_splits = apply_bipartite_ss_criterion(
        criterion,
        X[0],
        X[1],
        Y,
        start=[start_row, start_col],
        end=[end_row, end_col],
    )

    row_indices = np.arange(start_row+1, end_row) * n_cols
    col_indices = np.arange(start_col+1, end_col) * n_rows

    row_manual_splits = manually_eval_all_splits(
        x=x, y=y, supervision=supervision, indices=row_indices,
        start=start_row * n_cols,
        end=end_row * n_cols,
    )
    col_manual_splits = manually_eval_all_splits(
        x=xT, y=yT, supervision=supervision, indices=col_indices,
        start=start_col * n_rows,
        end=end_col * n_rows,
    )
    assert_equal_dicts(
        row_splits,
        row_manual_splits,
        msg_prefix='(rows) ',
        subset=row_manual_splits.keys(),
        ignore=['pos'],
        rtol=1e-4,
        atol=1e-8,
        differing_keys="raise",
    )
    assert_equal_dicts(
        col_splits,
        col_manual_splits,
        msg_prefix='(cols) ',
        subset=row_manual_splits.keys(),
        ignore=['pos'],
        rtol=1e-4,
        atol=1e-8,
        differing_keys="raise",
    )


def test_bipartite_ss_criterion_proxy_improvement(
    data, n_features, n_samples, supervision, random_state,
):
    start_row, start_col = 0, 0
    end_row, end_col = n_samples
    n_rows, n_cols = n_samples
    n_total_features = sum(n_features)
    feature = 1

    X, Y, x, y = data
    indices = X[0][:, feature].argsort()
    mono_indices = x[:, feature].argsort()

    X[0] = X[0][indices]
    Y = Y[indices]

    x = x[mono_indices]
    y = y[mono_indices]

    xT, yT = row_cartesian_product([X[1], X[0]]), Y.T.reshape(-1, 1)

    criterion = make_2dss_criterion(
        supervised_criteria=MSE,
        unsupervised_criteria=MSE,
        criterion_wrapper_class=SquaredErrorGSO,
        supervision=supervision,
        n_features=n_features,
        n_samples=n_samples,
        n_outputs=1,
        random_state=check_random_state(random_state),
        axis_decision_only=False,
    )
    mono_criterion = make_semisupervised_criterion(
        supervision=supervision,
        supervised_criterion=MSE,
        unsupervised_criterion=MSE,
        n_features=x.shape[1],
        n_samples=x.shape[0],
        n_outputs=y.shape[1],
    )

    row_splits, col_splits = apply_bipartite_ss_criterion(
        criterion,
        X[0],
        X[1],
        Y,
        start=[start_row, start_col],
        end=[end_row, end_col],
    )

    row_indices = np.arange(start_row+1, end_row) * n_cols
    col_indices = np.arange(start_col+1, end_col) * n_rows

    row_splits_mono = apply_criterion(
        mono_criterion,
        y=np.hstack((np.sqrt(n_total_features / n_features[0]) * x, y)),
        start=start_row * n_cols,
        end=end_row * n_cols,
    )
    col_splits_mono = apply_criterion(
        mono_criterion,
        # y=np.hstack((xT, yT)),
        y=np.hstack((np.sqrt(n_total_features / n_features[1]) * xT, yT)),
        start=start_col * n_rows,
        end=end_col * n_rows,
    )

    row_splits_mono_subsample = {
        k: v[row_indices-1] if isinstance(v, np.ndarray) else v
        for k, v in row_splits_mono.items()
    }
    col_splits_mono_subsample = {
        k: v[col_indices-1] if isinstance(v, np.ndarray) else v
        for k, v in col_splits_mono.items()
    }

    row_bipartite_proxy=row_splits['proxy_improvement']
    row_monopartite_proxy=row_splits_mono_subsample['proxy_improvement']
    row_corr = np.corrcoef(row_bipartite_proxy, row_monopartite_proxy)[0, 1]
    logging.info(f'{1-row_corr=}')

    col_bipartite_proxy=col_splits['proxy_improvement']
    col_monopartite_proxy=col_splits_mono_subsample['proxy_improvement']
    col_corr = np.corrcoef(col_bipartite_proxy, col_monopartite_proxy)[0, 1]
    logging.info(f'{1-col_corr=}')
    comparisons = [
        (  # 0
            row_splits_mono_subsample['proxy_improvement'],
            row_splits_mono_subsample['improvement'],
        ),
        (  # 1
            row_splits['proxy_improvement'],
            row_splits['improvement'],
        ),
        (  # 2
            row_splits['proxy_improvement'],
            row_splits_mono_subsample['proxy_improvement'],
        ),
        (  # 3
            row_splits['improvement'],
            row_splits_mono_subsample['proxy_improvement'],
        ),
        (  # 4
            col_splits_mono_subsample['proxy_improvement'],
            col_splits_mono_subsample['improvement'],
        ),
        (  # 5
            col_splits['proxy_improvement'],
            col_splits['improvement'],
        ),
        (  # 6
            col_splits['proxy_improvement'],
            col_splits_mono_subsample['proxy_improvement'],
        ),
        (  # 7
            col_splits['improvement'],
            col_splits_mono_subsample['proxy_improvement'],
        ),
    ]

    assert_equal_dicts(
        dict(corr=np.array([
            np.corrcoef(a, b)[0, 1] for a, b in comparisons
        ])[2]),
        dict(corr=1),
        rtol=1e-4,
    )
    return # XXX

    assert_equal_dicts(
        dict(corr=max(row_corr, col_corr)),
        dict(corr=1),
        rtol=1e-4,
        msg_prefix=f'Corr errors: {1 - row_corr:.7f} {1 - col_corr:.7f} | ',
    )
    # XXX
    # assert_equal_dicts(
    #     dict(corr=col_corr),
    #     dict(corr=1),
    #     rtol=1e-4,
    #     msg_prefix=f'Col corr error: {1 - col_corr:.7f} | ',
    # )
    # assert_equal_dicts(
    #     dict(corr=row_corr),
    #     dict(corr=1),
    #     rtol=1e-4,
    #     msg_prefix=f'Row corr error: {1 - row_corr:.7f} | ',
    # )

    # Compare proxy improvement values order
    top_proxies = 0  # consider all positions

    row_proxy_order = row_splits['proxy_improvement'].argsort()
    row_proxy_order_mono = row_splits_mono_subsample['proxy_improvement'].argsort()

    # consider only the top improvements, which matter the most for the splitter
    row_proxy_order = row_proxy_order[-top_proxies:]
    row_proxy_order_mono = row_proxy_order_mono[-top_proxies:]
    comparison_row_proxy = row_proxy_order == row_proxy_order_mono

    col_proxy_order = col_splits['proxy_improvement'].argsort()
    col_proxy_order_mono = col_splits_mono_subsample['proxy_improvement'].argsort()

    # consider only the top improvements, which matter the most for the splitter
    col_proxy_order = col_proxy_order[-top_proxies:]
    col_proxy_order_mono = col_proxy_order_mono[-top_proxies:]
    comparison_col_proxy = col_proxy_order == col_proxy_order_mono

    # XXX
    # bipartite_order = dict(
    #     bipartite_proxy=bipartite_proxy[row_proxy_order],
    #     monopartite_proxy=monopartite_proxy[row_proxy_order],
    # )
    # monopartite_order = dict(
    #     bipartite_proxy=bipartite_proxy[row_proxy_order_mono],
    #     monopartite_proxy=monopartite_proxy[row_proxy_order_mono],
    # )

    # assert_equal_dicts(
    #     bipartite_order,
    #     monopartite_order,
    #     msg_prefix=f'Corr. error: {1 - corr:.7f} | ',
    # )

    return  # XXX
    assert_allclose(
        row_splits['proxy_improvement'][row_proxy_order],
        row_splits['proxy_improvement'][row_proxy_order_mono],
        rtol=1e-4,
    )
    assert_allclose(
        row_splits_mono_subsample['proxy_improvement'][row_proxy_order],
        row_splits_mono_subsample['proxy_improvement'][row_proxy_order_mono],
        rtol=1e-4,
    )
    # =====

    assert comparison_row_proxy.all(), (
        f'(row proxy argsort) {row_proxy_order} {row_proxy_order_mono} = \n\t'
        + str(row_splits['proxy_improvement'][row_proxy_order]) + '\n\t'
        + str(row_splits_mono_subsample['proxy_improvement'][row_proxy_order_mono])
        + '\n-> '
        + comparison_text(
            row_splits['proxy_improvement'][row_proxy_order],
            row_splits['proxy_improvement'][row_proxy_order_mono],
            # row_proxy_order,
            # row_proxy_order_mono,
            comparison_row_proxy,
        )
    )

    # Since feature is on row_axis, proxies in cols are more likely to differ
    # assert comparison_col_proxy.all(), (
    #     f'(col proxy) {col_proxy_order} {col_proxy_order_mono} -> '
    #     + comparison_text(
    #         col_proxy_order,
    #         col_proxy_order_mono,
    #         comparison_col_proxy,
    #     )
    # )

    assert_equal_dicts(
        row_splits,
        row_splits_mono_subsample,
        msg_prefix='(rows 1d2d) ',
        ignore={
            'pos',
            'weighted_n_left',
            'weighted_n_right',
            'proxy_improvement',
        },
        rtol=1e-4,
        atol=1e-8,
        # differing_keys="raise",
    )
    assert_equal_dicts(
        col_splits,
        col_splits_mono_subsample,
        msg_prefix='(cols 1d2d) ',
        ignore={
            'pos',
            'weighted_n_left',
            'weighted_n_right',
            'proxy_improvement',
        },
        rtol=1e-4,
        atol=1e-8,
        # differing_keys="raise",
    )
    
    # Asserts bellow are mostly scaffold
    # ==================================
    row_manual_splits = manually_eval_all_splits(
        x=x, y=y, supervision=supervision, indices=row_indices,
        start=start_row * n_cols,
        end=end_row * n_cols,
    )
    col_manual_splits = manually_eval_all_splits(
        x=xT, y=yT, supervision=supervision, indices=col_indices,
        start=start_col * n_rows,
        end=end_col * n_rows,
    )

    row_manual_splits_complete = manually_eval_all_splits(
        x=x, y=y, supervision=supervision,
    )
    col_manual_splits_complete = manually_eval_all_splits(
        x=xT, y=yT, supervision=supervision,
    )

    assert_equal_dicts(
        row_splits,
        row_manual_splits,
        msg_prefix='(rows) ',
        subset=row_manual_splits.keys(),
        ignore=['pos'],
        rtol=1e-4,
        atol=1e-8,
        differing_keys="raise",
    )
    assert_equal_dicts(
        col_splits,
        col_manual_splits,
        msg_prefix='(cols) ',
        subset=row_manual_splits.keys(),
        ignore=['pos'],
        rtol=1e-4,
        atol=1e-8,
        differing_keys="raise",
    )
    assert_equal_dicts(
        row_splits_mono_subsample,
        row_manual_splits,
        msg_prefix='(rows mono) ',
        subset=row_manual_splits.keys(),
        ignore=['pos'],
        rtol=1e-4,
        atol=1e-8,
        differing_keys="raise",
    )
    assert_equal_dicts(
        col_splits_mono_subsample,
        col_manual_splits,
        msg_prefix='(cols mono) ',
        subset=row_manual_splits.keys(),
        ignore=['pos'],
        rtol=1e-4,
        atol=1e-8,
        differing_keys="raise",
    )

    assert_equal_dicts(
        row_splits_mono,
        row_manual_splits_complete,
        msg_prefix='(rows complete) ',
        rtol=1e-3,
        atol=1e-8,
    )
    assert_equal_dicts(
        col_splits_mono,
        col_manual_splits_complete,
        msg_prefix='(cols complete) ',
        rtol=1e-3,
        atol=1e-8,
    )
