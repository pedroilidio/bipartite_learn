import warnings
from numbers import Number
from argparse import ArgumentParser
from contextlib import contextmanager
from pprint import pformat
from time import time

import numpy as np
from .make_examples import make_interaction_data
from bipartite_learn.melter import row_cartesian_product

DTYPE_t, DOUBLE_t = np.float32, np.float64

# Default test params
DEF_PARAMS = dict(
    # seed=439,
    seed=7,
    shape=(50, 60),
    nattrs=(10, 9),
    nrules=5,
    min_samples_leaf=100,
    transpose_test=False,
    noise=0.1,
    inspect=False,
    plot=False,
)


@contextmanager
def stopwatch(msg=None):
    if msg:
        print(msg)
    t0 = time()
    yield
    t = time() - t0
    print(f"It took {t} s.")
    return t


def parse_args(**PARAMS):
    argparser = ArgumentParser(fromfile_prefix_chars='@')
    argparser.add_argument('--seed', type=int)
    argparser.add_argument('--shape', nargs='+', type=int)
    argparser.add_argument('--nattrs', nargs='+', type=int)
    argparser.add_argument('--nrules', type=int)
    argparser.add_argument('--min_samples_leaf', '--msl', type=int)

    argparser.add_argument('--transpose_test', action='store_true')
    argparser.add_argument('--noise', type=float)
    argparser.add_argument('--inspect', action='store_true')
    argparser.add_argument('--plot', action='store_true')

    for k in set(PARAMS) - set(DEF_PARAMS):
        v = PARAMS[k]
        argtype = str if (v is None) else type(v)
        argparser.add_argument('--'+k, type=argtype)

    PARAMS = DEF_PARAMS | PARAMS
    argparser.set_defaults(**PARAMS)

    return argparser.parse_args()


def gen_mock_data(melt=False, return_intervals=False, **PARAMS):
    with stopwatch("Generating mock interaction data with the following "
                   f"params:\n{pformat(PARAMS)}"):

        XX, Y, intervals = make_interaction_data(
            PARAMS['shape'], PARAMS['nattrs'], nrules=PARAMS['nrules'],
            noise=PARAMS['noise'], random_state=PARAMS['seed'],
            return_intervals=True,
        )

        if PARAMS['transpose_test']:
            print('Test transposing axis.')
            Y = np.copy(Y.T.astype(DOUBLE_t), order='C')
            XX = [np.ascontiguousarray(X, DTYPE_t) for X in XX[::-1]]
            PARAMS['nattrs'] = PARAMS['nattrs'][::-1]
            PARAMS['shape'] = PARAMS['shape'][::-1]
        else:
            Y = np.ascontiguousarray(Y, DOUBLE_t)
            XX = [np.ascontiguousarray(X, DTYPE_t) for X in XX]

    print('Data density (mean):', Y.mean())
    print('Data variance:', Y.var())
    print('=' * 50)

    ret = XX, Y

    if melt:
        ret += melt_2d_data(XX, Y)
    if return_intervals:
        ret += (intervals,)

    return ret


def melt_2d_data(XX, Y):
    return row_cartesian_product(XX), Y.reshape(-1, 1)


def float_comparison_text(a: float, b: float):
    a_str = f'{a:.8f}'
    b_str = f'{b:.8f}'

    if a_str == b_str:
        # otherwise the generator bellow will yield StopIteration
        return a_str

    diff_start = next(
        i for i, (ai, bi) in enumerate(zip(a_str, b_str)) if ai != bi
    )
    return f'{a_str[:diff_start]}({a_str[diff_start:]}!={b_str[diff_start:]})'


def comparison_text(a, b, equal_indices):
    if isinstance(a, float) and isinstance(b, float):
        return float_comparison_text(a, b)

    a_is_array = isinstance(a, np.ndarray) and a.ndim
    b_is_array = isinstance(b, np.ndarray) and b.ndim

    if not a_is_array and not b_is_array:
        return f"{a}{'==' if equal_indices else '!='}{b}"

    if a_is_array and b_is_array and a.shape != b.shape:
        return f"shape mismatch: {a.shape}!={b.shape}"

    n_diff = (~equal_indices).sum()
    if not a_is_array:
        a = np.repeat(a, b.size)
    elif not b_is_array:
        b = np.repeat(b, a.size)

    if n_diff == 0:
        return (
            np.array2string(a, edgeitems=2, threshold=5)
            + f' shape: {a.shape}'
        )

    a = a.reshape(-1)
    b = b.reshape(-1)

    return (
        f"{n_diff} differing elements out of {a.size} ({100*n_diff/a.size:.4g}%)."
        f"\n\tMax absolute diff: {np.abs(a-b).max()}"
        f"\n\tMax relative diff: {np.abs(1-a/b).max()}"
        f"\n\tdiff positions: {np.nonzero(~equal_indices)[0]}"
        "\n\tdiff: "
        + ' '.join(
            comparison_text(a[i], b[i], False)
            for i, is_eq in enumerate(equal_indices) if not is_eq
        )
    )


def assert_equal_dicts(
    d1: dict,
    d2: dict,
    ignore=None,
    subset=None,
    differing_keys="warn",  # warn, raise, or ignore
    msg_prefix: str = '',
    equal_nan: bool = False,
    rtol: float = 1e-7,
    atol: float = 0.0,
):
    ignore = set(ignore or [])
    keys1, keys2 = set(d1.keys()), set(d2.keys())
    keys = (keys1 | keys2)
    if subset:
        keys &= subset

    # keys that are not in both dicts neither in ignore
    unexpected_keys = (keys - (keys1 & keys2)) - ignore

    if differing_keys != "ignore" and unexpected_keys:
        msg = (
            f"Keys {unexpected_keys} are not present in both dictionaries "
            "nor listed in the 'ignore' parameter."
        )
        if differing_keys == "warn":
            warnings.warn(msg)
        elif differing_keys == "raise":
            raise ValueError(msg)
        else:
            raise NotImplementedError(f"Invalid {differing_keys=}")

    keys = sorted((keys - ignore) - unexpected_keys)
    equal_indices = []
    all_equal: list[bool] = []
    value_pairs = []
    names = []

    for key in keys:
        v1, v2 = d1[key], d2[key]

        names.append(key)
        value_pairs.append((v1, v2))

        if (
            not isinstance(v1, (Number, np.ndarray))
            and not isinstance(v2, (Number, np.ndarray))
        ):
            equal_indices.append(v1 == v2)
            all_equal.append(v1 == v2)
            continue

        if (
            isinstance(v1, np.ndarray)
            and isinstance(v2, np.ndarray)
            and v1.shape != v2.shape
        ):
            equal_indices.append(False)
            all_equal.append(False)
            continue

        eq_ind = np.isclose(
            v1, v2, rtol=rtol, atol=atol, equal_nan=equal_nan,
        )
        equal_indices.append(eq_ind)
        all_equal.append(eq_ind.all())

    differing_keys = [repr(n) for n, a in zip(names, all_equal) if not a]

    if differing_keys:
        text = differing_keys.pop()
    if differing_keys:
        text += ' and ' + differing_keys.pop()
    if differing_keys:
        text = ', '.join(differing_keys) + ', ' + text

    assert all(all_equal), (
        msg_prefix + text
        + ' values differ.\n'
        + '\n'.join(
            f"[{'OK' if a else 'X'}] {n}: {comparison_text(v[0], v[1], ii)}"
            for n, v, ii, a in zip(names, value_pairs, equal_indices, all_equal)
        )
    )
