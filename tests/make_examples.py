from argparse import ArgumentParser
from pathlib import Path
import numpy as np

DIR_HERE = Path(__file__).resolve().parent


def parse_args(args=None):
    argparser = ArgumentParser()
    argparser.add_argument('--shape', nargs='+', type=int, default=[1000, 800])
    argparser.add_argument('--nattrs', nargs='+', type=int, default=[200, 300])
    argparser.add_argument('--nrules', type=int, default=10)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('-o', '--outdir', type=Path,
                           default=Path(DIR_HERE.parent/'examples/input'))
    return argparser.parse_args(args)


def make_interaction_data(
         shape, nattrs, func=None, nrules=5, quiet=False, noise=0.,
         random_state=None,
 ):
    if not isinstance(random_state, np.random.Generator):
        random_state = np.random.default_rng(random_state)
    if func is None:
        func, strfunc = make_interaction_func(
            nattrs, nrules, random_state=random_state)
        if not quiet:
            print('Generated interaction function \n\t', strfunc)
    # shape contains the number of instances in each axis database, i.e.
    # its number of rows. nattrs contains their numbers of columns, i.e.
    # how many attributes per axis.
    XX = [random_state.random((ni, nj), dtype=np.float32)
          for ni, nj in zip(shape, nattrs)]
    # Create index tuples such as (np.newaxis, np.newaxis, :, np.newaxis).
    # That's because Y will be made usin numpy's broadcasting to explore
    # all combinations of x.
    ndim = len(shape)
    slices = np.tile(np.newaxis, (ndim, ndim)).astype(object)
    np.fill_diagonal(slices, slice(None))
    XXnewax = [X[tuple(s)] for X, s in zip(XX, slices)]
    # To ensure it is row-wise, not element-wise.
    # (a0),(a1),...,(a{ndim-1})->()
    sig = ','.join(f'(a{i})' for i in range(ndim)) + '->()'
    vfunc = np.vectorize(func, signature=sig)
    y = vfunc(*XXnewax).astype(int)
    if noise:
        y = y.astype(float)
        y += noise * random_state.random(y.shape)
    return XX, y, strfunc


def make_interaction_func(
        nattrs, nrules=10, popen=.5, pclose=.2, pand=.5,
        random_state=None,
):
    if not isinstance(random_state, np.random.Generator):
        random_state = np.random.default_rng(random_state)
    axes = random_state.choice(len(nattrs), nrules)
    attrs = [random_state.integers(nattrs[ax]) for ax in axes]
    cutoffs = random_state.random(nrules)
    orands = ['and ' if i else 'or '
              for i in random_state.random(nrules-1) < pand]
    orands.append('')

    strf = ''
    nopen = 0
    for ax, attr, cutoff, orand in zip(axes, attrs, cutoffs, orands):
       if random_state.random() < popen:
           strf += '( '
           nopen += 1
       strf += f'xx[{ax}][{attr}] < {cutoff} '
       if nopen and (random_state.random() < pclose):
           strf += ') '
           nopen -= 1
       strf += orand

    strf += ')' * nopen
    return eval('lambda *xx: ' + strf), strf


def main(shape, nattrs, nrules, outdir, seed):
    outdir.mkdir(exist_ok=True, parents=True)

    XX, Y, strfunc = make_interaction_data(
        shape, nattrs, func=func, random_state=seed
    )
    X1, X2 = XX

    with (outdir/'interaction_function.txt').open('w') as f:
        f.write(strfunc)

    print(f'Saving to {outdir.resolve()}...')
    np.savetxt(outdir/'X1.csv', X1, delimiter=',')
    np.savetxt(outdir/'X2.csv', X2, delimiter=',')
    np.savetxt(outdir/'Y.csv', Y, delimiter=',', fmt='%d')

    print('Generating labels...')
    X1_instance_labels = 'X1_instance_' + np.arange(shape[0]).astype(str).astype(object)
    X2_instance_labels = 'X2_instance_' + np.arange(shape[1]).astype(str).astype(object)
    X1_column_labels = 'X1_attr_' + np.arange(nattrs[0]).astype(str).astype(object)
    X2_column_labels = 'X2_attr_' + np.arange(nattrs[1]).astype(str).astype(object)

    print('Saving...')
    np.savetxt(outdir/'X1_names.txt', X1_instance_labels, fmt='%s')
    np.savetxt(outdir/'X2_names.txt', X2_instance_labels, fmt='%s')
    np.savetxt(outdir/'X1_col_names.txt', X1_column_labels, fmt='%s')
    np.savetxt(outdir/'X2_col_names.txt', X2_column_labels, fmt='%s')

    print('Done.')


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
