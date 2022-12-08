import warnings
from numbers import Number
from argparse import ArgumentParser
from contextlib import contextmanager
from pprint import pformat
from time import time

import numpy as np
from sklearn.utils._testing import assert_allclose
from make_examples import make_interaction_data
from hypertrees.melter import row_cartesian_product

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


def comparison_text(a, b, tol=1e-7):
    a_is_array = isinstance(a, np.ndarray)
    b_is_array = isinstance(b, np.ndarray)

    if (a_is_array and b_is_array and a.shape != b.shape):
        return f"shape mismatch: {a.shape} != {b.shape}"

    differing = np.abs(a - b) >= tol

    if a_is_array or b_is_array:
        n_diff = differing.sum()
        if n_diff == 0:
            return (
                f"{np.array2string(a, edgeitems=2, threshold=5) if a_is_array else a} "
                f"== {np.array2string(b, edgeitems=2, threshold=5) if b_is_array  else b}"
            )
        return (
            f"{n_diff} differing elements."
            f"\n\tdiff positions: {np.nonzero(differing)[0]}"
            "\n\tdiff: "
            + ', '.join(
                f"{a.reshape(-1)[i] if a_is_array else a}"
                f"|{b.reshape(-1)[i] if b_is_array else b}"
                for i, is_diff in enumerate(differing) if is_diff
            )
        )
    return f"{a} {'!=' if differing else '=='} {b}"


def assert_equal_dicts(d1: dict, d2: dict, ignore=None, warn=False):
    ignore = ignore or set()
    keys = {*d1.keys(), *d2.keys()} - set(ignore)
    assertions = []
    value_pairs = []
    names = []

    for key in keys:
        if key not in d1:
            if warn:
                warnings.warn(f"Key {key!r} not in first dict.")
            continue
        if key not in d2:
            if warn:
                warnings.warn(f"Key {key!r} not in second dict.")
            continue

        v1, v2 = d1[key], d2[key]

        if (
            not isinstance(v1, (Number, np.ndarray))
            or not isinstance(v2, (Number, np.ndarray))
        ):
            warnings.warn(
                f"{key!r} not numeric or array attribute (values: {v1} {v2})."
            )
            continue

        names.append(key)
        value_pairs.append((v1, v2))
        # assertions.append(abs(v1-v2) < tol)
        try:
            assert_allclose(v1, v2, atol=1e-7, verbose=False)
            assertions.append(None)
        except AssertionError as e:
            assertions.append(e)
        
    differing = [repr(n) for n, a in zip(names, assertions) if a is not None]

    if differing:
        text = differing.pop()
    if differing:
        text += ' and ' + differing.pop()
    if differing:
        text = ', '.join(differing) + ', ' + text

    assert all(A is None for A in assertions), (
        text
        + ' values differ.\n'
        + '\n'.join(
            f"{n}: {comparison_text(v[0], v[1])}"  # \n{a}"
            for n, v, a in zip(names, value_pairs, assertions)
        )
    )
 