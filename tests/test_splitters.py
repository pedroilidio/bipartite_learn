from typing import Callable
import numpy as np
from functools import partial

from sklearn.tree._criterion import MSE
from sklearn.tree._splitter import BestSplitter

from hypertrees.tree._nd_splitter import make_2d_splitter
from hypertrees.tree._semisupervised_criterion import (
    SSCompositeCriterion,
    make_2dss_splitter,
    SingleFeatureSSCompositeCriterion,
    MSE2DSFSS,
    SFSSMSE,
)
from hypertrees.tree._semisupervised_splitter import BestSplitterSFSS
from hypertrees.tree._experimental_criterion import UD3, UD35

from splitter_test import test_splitter, test_splitter_nd
from test_utils import (
    parse_args, stopwatch, gen_mock_data, melt_2d_data,
)

#from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64

from pathlib import Path
from time import time
from pprint import pprint
import warnings

# Default test params
DEF_PARAMS = dict(
    seed=0,
    shape=(50, 60),
    nattrs=(10, 9),
    nrules=1,
    min_samples_leaf=100,
    transpose_test=False,
    noise=.5,
    inspect=False,
    plot=False,
    start=0,
    end=0,
    supervision=-1.,
)


def compare_splitters_1d2d_ideal(
    splitter1,
    splitter2,
    tol=0,
    **PARAMS,
):
    PARAMS = DEF_PARAMS | dict(noise=0) | PARAMS

    if PARAMS['noise']:
        warnings.warn(f"noise={PARAMS['noise']}. Setting it to zero"
                      " since noise=0 is what defines an ideal split.")
        PARAMS['noise'] = 0
    
    result1, result2 = compare_splitters_1d2d(
        splitter1, splitter2, tol, **PARAMS)

    assert result1['improvement'] != 0
    assert result1['impurity_left'] == 0
    assert result1['impurity_right'] == 0

    assert result2['improvement'] != 0
    assert result2['impurity_left'] == 0
    assert result2['impurity_right'] == 0


def compare_splitters_1d2d(
    splitter1,
    splitter2,
    semisupervised_1d=False,
    unsupervised_1d=False,
    single_feature_ss_1d=False,
    only_1d=False,
    tol=1e-10,
    manual_impurity=True,
    **PARAMS,
):
    PARAMS = DEF_PARAMS | PARAMS
    if not isinstance(PARAMS['seed'], np.random.RandomState):
        random_state = np.random.RandomState(PARAMS['seed'])

    # XX, Y, x, y, _ = gen_mock_data(**PARAMS, melt=True)
    XX, Y, _ = gen_mock_data(**PARAMS)
    n_samples_0 = PARAMS['shape'][0]

    # Artificially set XX[0][:, 0] to be the best (most separable)
    # unsupervised attribute. 
    XX[0][:, 0] = np.hstack([
        random_state.normal(loc=.2, scale=.1, size=n_samples_0//2),
        random_state.normal(loc=.8, scale=.1, size=n_samples_0-n_samples_0//2),
    ])
    x, y = melt_2d_data(XX, Y)

    if Y.var() == 0:
        raise RuntimeError(f"Bad seed ({PARAMS['seed']}), y is homogeneus."
                           " Try another one or reduce nrules.")

    start = PARAMS['start'] or 0
    end = PARAMS['end'] or Y.shape[0]

    if single_feature_ss_1d:
        x = x[:, 0].reshape(-1, 1)
    if semisupervised_1d:
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
            min_samples_leaf=y.shape[1],
            min_weight_leaf=0,
            random_state=np.random.RandomState(PARAMS['seed']),
        )

    if isinstance(splitter2, Callable):
        if only_1d:
            splitter2 = splitter2(
                criterion=MSE(n_outputs=y.shape[1], n_samples=x.shape[0]),
                max_features=x.shape[1],
                min_samples_leaf=y.shape[1],
                min_weight_leaf=0,
                random_state=random_state,
            )
        else:
            splitter2 = make_2d_splitter(
                splitters=splitter2,
                criteria=MSE,
                max_features=[X.shape[1] for X in XX],
                n_samples=Y.shape,
                n_outputs=1,
                random_state=random_state,
            )

    # Run test
    with stopwatch(f'Testing 1D splitter ({splitter1.__class__.__name__})...'):
        result1 = test_splitter(
            splitter1, x, y, start=start*Y.shape[1], end=end*Y.shape[1])
        print('Best split found:')
        pprint(result1)

    assert result1['improvement'] >= 0, \
        'Negative reference improvement, input seems wrong.'

    if not semisupervised_1d and manual_impurity:
        x_ = x[start*Y.shape[1] : end*Y.shape[1]]
        y_ = y[start*Y.shape[1] : end*Y.shape[1]]
        pos = result1['pos'] - start*Y.shape[1]

        sorted_indices = x_[:, result1['feature']].argsort()
        manual_impurity_left = y_[sorted_indices][:pos].var(0).mean()
        manual_impurity_right = y_[sorted_indices][pos:].var(0).mean()
        print(f'* manual_impurity_left={manual_impurity_left}')
        print(f'* manual_impurity_right={manual_impurity_right}')

        assert abs(result1['impurity_left']-manual_impurity_left) <= tol, \
            ('Wrong reference impurity left: '
             f'{result1["impurity_left"]} != {manual_impurity_left}')
        assert abs(result1['impurity_right']-manual_impurity_right) <= tol, \
            ('Wrong reference impurity right: '
             f'{result1["impurity_left"]} != {manual_impurity_left}')

    # Run test 2d
    with stopwatch(f'Testing 2D splitter ({splitter2.__class__.__name__})...'):
        if only_1d:
            result2 = test_splitter(
                splitter2, x, y, start=start*Y.shape[1], end=end*Y.shape[1])
        else:
            result2 = test_splitter_nd(
                splitter2, XX, Y, start=[start, 0], end=[end, Y.shape[1]])
        print('Best split found:')
        pprint(result2)

    assert result2['threshold'] == result1['threshold'], \
        'threshold differs from reference.'
    assert abs(result2['improvement']-result1['improvement']) <= tol, \
        'improvement differs from reference.'
    assert abs(result2['impurity_left']-result1['impurity_left']) <= tol, \
        'impurity_left differs from reference.'
    assert abs(result2['impurity_right']-result1['impurity_right']) <= tol, \
        'impurity_right differs from reference.'
    
    return result1, result2


def test_1d2d_ideal(**PARAMS):
    print('*** test_1d2d_ideal')
    PARAMS = DEF_PARAMS | PARAMS
    return compare_splitters_1d2d_ideal(
        splitter1=BestSplitter,
        splitter2=BestSplitter,
        **PARAMS,
    )


def test_1d2d(**PARAMS):
    print('*** test_1d2d')
    PARAMS = DEF_PARAMS | PARAMS
    return compare_splitters_1d2d(
        splitter1=BestSplitter,
        splitter2=BestSplitter,
        **PARAMS,
    )


def test_ss_1d2d_sup(**PARAMS):
    print('*** test_ss_1d2d_sup')
    PARAMS = DEF_PARAMS | PARAMS
    ss2d_splitter=make_2dss_splitter(
        splitters=BestSplitter,
        criteria=MSE,
        supervision=1.,
        max_features=PARAMS['nattrs'],
        n_features=PARAMS['nattrs'],
        n_samples=PARAMS['shape'],
        n_outputs=1,
    )

    return compare_splitters_1d2d(
        splitter1=BestSplitter,
        splitter2=ss2d_splitter,
        **PARAMS,
    )


def test_ss_1d2d_unsup(**PARAMS):
    print('*** test_ss_1d2d_unsup')
    PARAMS = DEF_PARAMS | PARAMS
    ss2d_splitter=make_2dss_splitter(
        splitters=BestSplitter,
        criteria=MSE,
        supervision=0.,
        max_features=PARAMS['nattrs'],
        n_features=PARAMS['nattrs'],
        n_samples=PARAMS['shape'],
        n_outputs=1,
    )

    return compare_splitters_1d2d(
        splitter1=BestSplitter,
        splitter2=ss2d_splitter,
        unsupervised_1d=True,
        **PARAMS,
    )


def test_ss_1d2d(**PARAMS):
    """Compare 1D to 2D version of semisupervised MSE splitter.
    """
    print('*** test_ss_1d2d')
    PARAMS = DEF_PARAMS | PARAMS
    rstate = np.random.RandomState(PARAMS['seed'])
    supervision = PARAMS.get('supervision', -1.)
    if supervision == -1.:
        supervision = rstate.random()
    print(f"* Set supervision={supervision}")

    splitter1 = BestSplitter(
        criterion=SSCompositeCriterion(
            supervision=supervision,
            criterion=MSE,
            n_features=np.sum(PARAMS['nattrs']),
            n_samples=np.prod(PARAMS['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(PARAMS['nattrs']),
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=rstate,
    )

    ss2d_splitter=make_2dss_splitter(
        splitters=BestSplitter,
        criteria=MSE,
        supervision=supervision,
        max_features=PARAMS['nattrs'],
        n_features=PARAMS['nattrs'],
        n_samples=PARAMS['shape'],
        n_outputs=1,
        random_state=rstate,
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
    )

    return compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=ss2d_splitter,
        semisupervised_1d=True,
        **PARAMS,
    )


def test_ss_1d2d_ideal_split(**PARAMS):
    print('*** test_ss_1d2d_ideal_split')
    PARAMS = DEF_PARAMS | PARAMS
    ss2d_splitter=make_2dss_splitter(
        splitters=BestSplitter,
        criteria=MSE,
        supervision=1.,
        max_features=PARAMS['nattrs'],
        n_features=PARAMS['nattrs'],
        n_samples=PARAMS['shape'],
        n_outputs=1,
    )

    return compare_splitters_1d2d_ideal(
        splitter1=BestSplitter,
        splitter2=ss2d_splitter,
        **PARAMS,
    )


def test_sfss_1d_sup(**PARAMS):
    print('*** test_sfss_1d_sup')
    PARAMS = DEF_PARAMS | PARAMS
    rstate = np.random.RandomState(PARAMS['seed'])

    splitter1 = BestSplitterSFSS(
        criterion=SingleFeatureSSCompositeCriterion(
            supervision=1.,
            criterion=MSE,
            n_features=np.sum(PARAMS['nattrs']),
            n_samples=np.prod(PARAMS['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(PARAMS['nattrs']),
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=rstate,
    )

    return compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=BestSplitter,
        **PARAMS,
    )


def test_sfss_1d_unsup(**PARAMS):
    print('*** test_sfss_1d_unsup')
    PARAMS = DEF_PARAMS | PARAMS
    rstate = np.random.RandomState(PARAMS['seed'])

    splitter1 = BestSplitterSFSS(
        criterion=SingleFeatureSSCompositeCriterion(
            supervision=0.,
            criterion=MSE,
            n_features=np.sum(PARAMS['nattrs']),
            n_samples=np.prod(PARAMS['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(PARAMS['nattrs']),
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=rstate,
    )

    return compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=BestSplitter,
        single_feature_ss_1d=True,
        only_1d=True,
        **PARAMS,
    )


def test_sfss_2d_sup(**PARAMS):
    print('*** test_sfss_2d_sup')
    PARAMS = DEF_PARAMS | PARAMS
    rstate = np.random.RandomState(PARAMS['seed'])

    ss2d_splitter=make_2dss_splitter(
        splitters=BestSplitterSFSS,
        criteria=MSE,
        ss_criteria=SingleFeatureSSCompositeCriterion,
        supervision=1.,
        max_features=PARAMS['nattrs'],
        n_features=1,
        n_samples=PARAMS['shape'],
        n_outputs=1,
        random_state=rstate,
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
    )

    return compare_splitters_1d2d(
        splitter1=BestSplitter,
        splitter2=ss2d_splitter,
        **PARAMS,
    )


# FIXME
def test_sfss_2d_unsup(**PARAMS):
    print('*** test_sfss_2d_unsup')
    PARAMS = DEF_PARAMS | PARAMS
    rstate = np.random.RandomState(PARAMS['seed'])

    ss2d_splitter=make_2dss_splitter(
        splitters=BestSplitterSFSS,
        criteria=MSE,
        ss_criteria=SingleFeatureSSCompositeCriterion,
        supervision=0.,
        max_features=PARAMS['nattrs'],
        n_features=1,
        n_samples=PARAMS['shape'],
        n_outputs=1,
        random_state=rstate,
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
    )

    return compare_splitters_1d2d(
        splitter1=BestSplitter,
        splitter2=ss2d_splitter,
        single_feature_ss_1d=True,
        unsupervised_1d=True,
        **PARAMS,
    )


def test_sfss_1d2d(**PARAMS):
    """Compare 1D to 2D version of semisupervised MSE splitter.
    """
    print('*** test_sfss_1d2d')
    PARAMS = DEF_PARAMS | PARAMS
    rstate = np.random.RandomState(PARAMS['seed'])
    supervision = PARAMS.get('supervision', -1.)
    if supervision == -1.:
        supervision = rstate.random()
    print(f"* Set supervision={supervision}")

    splitter1 = BestSplitterSFSS(
        criterion=SingleFeatureSSCompositeCriterion(
            supervision=supervision,
            criterion=MSE,
            n_features=1,
            n_samples=np.prod(PARAMS['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(PARAMS['nattrs']),
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=rstate,
    )

    ss2d_splitter=make_2dss_splitter(
        splitters=BestSplitterSFSS,
        criteria=MSE,
        ss_criteria=SingleFeatureSSCompositeCriterion,
        supervision=supervision,
        max_features=PARAMS['nattrs'],
        n_features=1,
        n_samples=PARAMS['shape'],
        n_outputs=1,
        random_state=rstate,
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
    )

    return compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=ss2d_splitter,
        semisupervised_1d=True,
        **PARAMS,
    )


def test_ud3_1d2d_unsup(**PARAMS):
    """Compare 1D to 2D version of semisupervised MSE splitter.
    """
    print('*** test_ud3_1d2d_unsup')
    PARAMS = DEF_PARAMS | PARAMS
    rstate = np.random.RandomState(PARAMS['seed'])

    splitter1 = BestSplitterSFSS(
        criterion=SingleFeatureSSCompositeCriterion(
            supervision=0.,
            supervised_criterion=MSE,
            unsupervised_criterion=UD3,
            n_features=1,
            n_samples=np.prod(PARAMS['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(PARAMS['nattrs']),
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=rstate,
    )

    ss2d_splitter=make_2dss_splitter(
        splitters=BestSplitterSFSS,
        ss_criteria=SingleFeatureSSCompositeCriterion,
        supervised_criteria=MSE,
        unsupervised_criteria=UD3,
        supervision=0.,
        max_features=PARAMS['nattrs'],
        n_features=1,
        n_samples=PARAMS['shape'],
        n_outputs=1,
        random_state=rstate,
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
    )

    result1, result2 = compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=ss2d_splitter,
        # unsupervised_1d=True,
        semisupervised_1d=True,
        manual_impurity=False,
        **PARAMS,
    )

    assert result1['feature'] == 0
    assert result2['feature'] == 0


def test_ud35_1d2d_unsup(**PARAMS):
    """Compare 1D to 2D version of semisupervised MSE splitter.
    """
    print('*** test_ud35_1d2d_unsup')
    PARAMS = DEF_PARAMS | PARAMS
    rstate = np.random.RandomState(PARAMS['seed'])

    splitter1 = BestSplitterSFSS(
        criterion=SingleFeatureSSCompositeCriterion(
            supervision=0.,
            supervised_criterion=MSE,
            unsupervised_criterion=UD35,
            n_features=1,
            n_samples=np.prod(PARAMS['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(PARAMS['nattrs']),
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=rstate,
    )

    ss2d_splitter=make_2dss_splitter(
        splitters=BestSplitterSFSS,
        ss_criteria=SingleFeatureSSCompositeCriterion,
        supervised_criteria=MSE,
        unsupervised_criteria=UD35,
        supervision=0.,
        max_features=PARAMS['nattrs'],
        n_features=1,
        n_samples=PARAMS['shape'],
        n_outputs=1,
        random_state=rstate,
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
    )

    result1, result2 = compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=ss2d_splitter,
        # unsupervised_1d=True,
        semisupervised_1d=True,
        manual_impurity=False,
        **PARAMS,
    )

    assert result1['feature'] == 0
    assert result2['feature'] == 0


def test_sfssmse_1d(**PARAMS):
    """Compare 1D to 2D version of semisupervised MSE splitter.
    """
    print('*** test_sfssmse_1d')
    PARAMS = DEF_PARAMS | PARAMS
    rstate = np.random.RandomState(PARAMS['seed'])
    supervision = PARAMS.get('supervision', -1.)
    if supervision == -1.:
        supervision = rstate.random()
    print(f"* Set supervision={supervision}")

    splitter1 = BestSplitterSFSS(
        criterion=SFSSMSE(
            supervision=supervision,
            n_features=1,
            n_samples=np.prod(PARAMS['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(PARAMS['nattrs']),
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=rstate,
    )

    ss2d_splitter=make_2dss_splitter(
        splitters=BestSplitterSFSS,
        ss_criteria=SingleFeatureSSCompositeCriterion,
        criteria=MSE,
        supervision=supervision,
        max_features=PARAMS['nattrs'],
        n_features=1,
        n_samples=PARAMS['shape'],
        n_outputs=1,
        random_state=rstate,
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
    )

    return compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=ss2d_splitter,
        semisupervised_1d=True,
        **PARAMS,
    )


def test_sfssmse_1d2d(**PARAMS):
    """Compare 1D to 2D version of semisupervised MSE splitter.
    """
    print('*** test_sfssmse_1d2d')
    PARAMS = DEF_PARAMS | PARAMS
    rstate = np.random.RandomState(PARAMS['seed'])
    supervision = PARAMS.get('supervision', -1.)
    if supervision == -1.:
        supervision = rstate.random()
    print(f"* Set supervision={supervision}")

    splitter1 = BestSplitterSFSS(
        criterion=SFSSMSE(
            supervision=supervision,
            n_features=1,
            n_samples=np.prod(PARAMS['shape']),
            n_outputs=1,
        ),
        max_features=np.sum(PARAMS['nattrs']),
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        random_state=rstate,
    )

    ss2d_splitter=make_2dss_splitter(
        splitters=BestSplitterSFSS,
        ss_criteria=SFSSMSE,
        supervision=supervision,
        max_features=PARAMS['nattrs'],
        n_features=1,
        n_samples=PARAMS['shape'],
        n_outputs=1,
        random_state=rstate,
        min_samples_leaf=PARAMS['min_samples_leaf'],
        min_weight_leaf=0.,
        criterion_wrapper_class=MSE2DSFSS,
    )

    return compare_splitters_1d2d(
        splitter1=splitter1,
        splitter2=ss2d_splitter,
        semisupervised_1d=True,
        **PARAMS,
    )


# TODO: test with different axis supervisions. sscrit2d.impurity_improvement()
#       may fail.
def main(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    test_1d2d_ideal(**vars(args))
    test_1d2d(**vars(args))
    test_ss_1d2d_sup(**vars(args))
    test_ss_1d2d_unsup(**vars(args))
    test_ss_1d2d(**vars(args))
    test_ss_1d2d_ideal_split(**vars(args))
    test_sfss_1d_sup(**vars(args))
    test_sfss_1d_unsup(**vars(args))
    test_sfss_2d_sup(**vars(args))
    test_sfss_2d_unsup(**vars(args))
    test_sfss_1d2d(**vars(args))
    test_sfssmse_1d(**vars(args))
    test_sfssmse_1d2d(**vars(args))
    # test_ud3_1d2d_unsup(**vars(args))  # FIXME
    # test_ud35_1d2d_unsup(**vars(args))  # FIXME


if __name__ == "__main__":
    args = parse_args(**DEF_PARAMS)
    main(**vars(args))