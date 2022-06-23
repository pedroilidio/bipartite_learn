from make_examples import make_interaction_data
from hypertrees.melter import row_cartesian_product

from time import time
from argparse import ArgumentParser
from contextlib import contextmanager
import numpy as np
from pprint import pformat

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


def gen_mock_data(melt=False, **PARAMS):
    with stopwatch("Generating mock interaction data with the following "
                   f"params:\n{pformat(PARAMS)}"):

        XX, Y, strfunc = make_interaction_data(
            PARAMS['shape'], PARAMS['nattrs'], nrules=PARAMS['nrules'],
            noise=PARAMS['noise'], random_state=PARAMS['seed']
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

    if melt:
        return XX, Y, *melt_2d_data(XX, Y), strfunc
    else:
        return XX, Y, strfunc


def melt_2d_data(XX, Y):
    return row_cartesian_product(XX), Y.reshape(-1, 1)
