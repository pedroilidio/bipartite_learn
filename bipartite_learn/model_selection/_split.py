import itertools
import copy
from numbers import Integral
import numpy as np
from sklearn.utils._param_validation import validate_params, Interval
from sklearn.model_selection._split import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
    _validate_shuffle_split,
)
from sklearn.utils.validation import _num_samples

from bipartite_learn.datasets.base import BipartiteDataset


class MultipartiteCrossValidator(BaseCrossValidator):
    @validate_params(
        {
            "cross_validators": [BaseCrossValidator, list, tuple],
            "diagonal": ["boolean"],
            "n_parts": [Interval(Integral, 2, None, closed="left")],
        }
    )
    def __init__(self, cross_validators, diagonal=False, n_parts=2):
        if isinstance(cross_validators, BaseCrossValidator):
            self.cross_validators = [
                copy.deepcopy(cross_validators) for _ in range(n_parts)
            ]
            self.n_parts = n_parts
        else:
            self.n_parts = len(cross_validators)
            self.cross_validators = cross_validators

        self.diagonal = diagonal

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._check_n_splits(X, y, groups)

    def _check_n_splits(self, X=None, y=None, groups=None):
        ax_n_splits = list(
            self._get_from_children_cv("get_n_splits", X, y, groups),
        )

        if self.diagonal:
            # Ensure all cross validators report the same number of splits.
            g = itertools.groupby(ax_n_splits)
            next(g, None)

            # if not all elements in ax_n_splits are equal:
            if next(g, False):
                raise ValueError(
                    "Cross-validators must generate the same number"
                    " of splits if diagonal=True"
                )
            return ax_n_splits[0]

        return np.prod(ax_n_splits)

    def split(self, X, y=None, groups=None):
        train_test = self._get_from_children_cv("split", X, y=y, groups=groups)
        # Each train_test element represents a fold, containing train/test
        # indices for all axes:
        #
        # next(train_test) == (
        #    (train_fold0_ax0, test_fold0_ax0),
        #    (train_fold0_ax1, test_fold0_ax1),
        #    (train_fold0_ax2, test_fold0_ax2),
        #    ...
        # ),
        # train_test.shape = n_folds, n_axis, 2

        # So train_test is
        #   (all_train/test_tuples for fold1,
        #    all_train/test_tuples for fold2,

        combine_func = zip if self.diagonal else itertools.product
        split_iter = combine_func(*train_test)
        # split_iter.shape = n_axis, 2, n_folds

        # For each axis, pick one train/test tuple from a diferent fold.
        # next(splits_iter) == (
        #   ((train_ax0_fold_a, test_ax0_fold_a),
        #    (train_ax1_fold_c, test_ax1_fold_c),
        #    (train_ax2_fold_h, test_ax2_fold_h),
        #    (train_ax3_fold_b, test_ax3_fold_b),
        #    ...
        #   ),

        # Put in another way:
        # next(splits_iter) == (train_test_ax1, train_test_ax2, ...)

        for split in split_iter:
            yield split

    def _iter_test_masks(self, X=None, y=None, groups=None):
        iter_ = self._get_from_children_cv(
            "_iter_test_masks",
            X,
            y=y,
            groups=groups,
        )

        for item in iter_:
            yield item

    def _iter_test_indices(self, X=None, y=None, groups=None):
        iter_ = self._get_from_children_cv(
            "_iter_test_indices",
            X,
            y=y,
            groups=groups,
        )

        for item in iter_:
            yield item

    def _get_from_children_cv(self, attr, X=None, y=None, groups=None):
        if X is None:
            X = [None] * self.n_parts
        if groups is None:
            groups = [None] * self.n_parts
        if y is None:
            y = [None] * self.n_parts
        else:
            # FIXME: is np.moveaxis well suited here?
            y = [np.moveaxis(y, ax, 0) for ax in range(self.n_parts)]

        for cv, Xi, yi, group in zip(self.cross_validators, X, y, groups):
            yield getattr(cv, attr)(Xi, yi, group)


class HoldOut(BaseCrossValidator):
    def __init__(self, test_size=None, train_size=None):
        self.test_size = test_size
        self.train_size = train_size
        self._default_test_size = 0.25

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )
        yield np.arange(n_train, n_samples)


def _check_train_test_combinations(
    ttc,
    n_parts,
    include_train=None,
    symbols="LT",
    sep="",
):
    """Check train_test_combinations parameter."""
    # ttc = train_test_combinations
    include_train = include_train or True
    names_provided = (ttc is not None) and (
        isinstance(ttc[0], str) or isinstance(ttc[0][0], str)
    )

    if ttc is None:
        ret_ttc = itertools.product((0, 1), repeat=n_parts)
        if not include_train:
            next(ret_ttc)  # Skip (0, 0, ...), only training data.
        ret_ttc = list(ret_ttc)

    elif any(len(c) != 2 for c in ttc):
        raise ValueError(
            "train_test_combinations must contain only 2-lengthed sequences."
        )

    # If ttc contains string or list-like of strings, translate.
    elif names_provided:
        names = ttc
        ret_ttc = [[symbols.index(s) for s in train_test] for train_test in names]

    if not names_provided:
        names = [[symbols[is_test] for is_test in train_test] for train_test in ret_ttc]

    names = [n[0] + sep + n[1] for n in names]
    return ret_ttc, names


def _repeat_if_single(n_parts, *args):
    result = []
    for arg in args:
        if not isinstance(arg, (tuple, list)):  # TODO: Better criteria?
            arg = [copy.deepcopy(arg) for _ in range(n_parts)]
        result.append(arg)
    return result


def combine_train_test_indices(
    train_test,  # iterable of (train_indices, test_indices) tuples for each axis.
    train_test_combinations=None,
    include_train=None,
):
    n_parts = len(train_test)

    train_test_combinations, train_test_names = _check_train_test_combinations(
        ttc=train_test_combinations,
        n_parts=n_parts,
        include_train=include_train,
    )

    train_test_index_combinations = {}

    # NOTE: ttc stands for train-test combinations
    for is_test_tuple, ttc_name in zip(train_test_combinations, train_test_names):
        # is_test_tuple ~= (0, 1, 1, 0)
        test_indices = [
            ax_train_test[is_test]
            for is_test, ax_train_test in zip(is_test_tuple, train_test)
        ]
        train_test_index_combinations[ttc_name] = test_indices

    return train_test_index_combinations


def make_multipartite_kfold(
    cv=None,
    shuffle=False,
    stratified=False,
    random_state=None,
    diagonal=False,
    n_parts=None,
):
    """Build a n_parts-partite KFold cross-validator.

    Wraps n_parts KFold or StratifiedKFold cross-validators with a
    MultipartiteCrossValidator, that provides train-test indices for splitting
    InputDataND objects. The returned wrapper object is compatible with
    parameter search objects in bipartite_learn.model_selection, such as
    MultipartiteGridSearchCV and MultipartiteRandomizedSearchCV.

    The parameters 'cv', 'shuffle' and 'stratified' can be
    optionally passed as a tuple or list of values intended for each respective
    axis of the ND data. Otherwise, the single value is equally considered for
    all axes.

    Parameters
    ----------
    cv : int or CrossValidator or list-like of these, default=None
        TODO
        One can optionally pass values for each axis in a list or tuple. If a
        single value is provided, it will be considered for all axes.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    shuffle : bool or list-like of bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
        One can optionally pass values for each axis in a list or tuple. If a
        single value is provided, it will be considered for all axes.
    stratified : bool or list-like of array-like, default=None
        whether or not to generate a stratified splitter.
        Read more in the :ref:`User Guide <stratification>`.
        One can optionally pass arrays for each axis in a list or tuple. If a
        single array is provided, it will be considered for all axes.
    n_parts : int
        If none of the other parameters is a list or a sequence, you must indi-
        cate the number of dimensions.

    Returns
    -------
    cross-validation splitter : MultipartiteCrossValidator
    """
    if n_parts is None:
        if isinstance(cv, (list, tuple)):
            n_parts = len(cv)
        elif isinstance(shuffle, (list, tuple)):
            n_parts = len(shuffle)
        elif isinstance(stratified, (list, tuple)):
            n_parts = len(stratified)
        else:
            raise ValueError(
                "If none of cv, shuffle or stratified is tuple"
                " or list, n_parts must be provided."
            )

    (
        cv,
        shuffle,
        stratified,
    ) = _repeat_if_single(n_parts, cv, shuffle, stratified)

    cv_list = []

    for ax in range(n_parts):
        if not isinstance(cv[ax], BaseCrossValidator):
            CVClass = StratifiedKFold if stratified[ax] else KFold
            cv[ax] = CVClass(
                n_splits=cv[ax],
                random_state=random_state,
                shuffle=shuffle[ax],
            )
        cv_list.append(copy.deepcopy(cv[ax]))

    return check_multipartite_cv(cv_list, n_parts=n_parts, diagonal=diagonal)


def make_multipartite_train_test_splitter(
    n_parts,
    test_size=None,
    train_size=None,
    shuffle=False,
    stratified=False,
    random_state=None,
):
    """Build a len(n_samples)-dimensional train-test splitter.

    Wraps len(n_samples) CrossValidator data splitters with a
    MultipartiteCrossValidator, that provides train-test indices for splitting
    InputDataND objects. The returned wrapper object is compatible with
    parameter search objects in bipartite_learn.model_selection, such as
    MultipartiteGridSearchCV and MultipartiteRandomizedSearchCV.

    The parameters 'test_size', 'train_size', 'shuffle' and 'stratify' can be
    optionally passed as a tuple or list of values intended for each respective
    axis of the ND data. Otherwise, the single value is equally considered for
    all axes.

    Parameters
    ----------
    test_size : float or int or list-like of (float or int), default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
        One can optionally pass values for each axis in a list or tuple. If a
        single value is provided, it will be considered for all axes.
    train_size : float or int or list-like of (float or int), default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
        One can optionally pass values for each axis in a list or tuple. If a
        single value is provided, it will be considered for all axes.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    shuffle : bool or list-like of bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
        One can optionally pass values for each axis in a list or tuple. If a
        single value is provided, it will be considered for all axes.
    stratified : bool or list-like of array-like, default=None
        whether or not to generate a stratified splitter.
        Read more in the :ref:`User Guide <stratification>`.
        One can optionally pass arrays for each axis in a list or tuple. If a
        single array is provided, it will be considered for all axes.

    Returns
    -------
    cross-validation splitter : MultipartiteCrossValidator
    """
    (
        test_size,
        train_size,
        shuffle,
        stratified,
    ) = _repeat_if_single(n_parts, test_size, train_size, shuffle, stratified)

    cv_list = []

    for ax in range(n_parts):
        if shuffle[ax]:
            if stratified[ax]:
                CVClass = StratifiedShuffleSplit
            else:
                CVClass = ShuffleSplit

            cv = CVClass(
                test_size=test_size[ax],
                train_size=train_size[ax],
                random_state=random_state,
                n_splits=1,
            )
        else:
            if stratified[ax]:
                raise ValueError(
                    "Stratified train/test split is not implemented for"
                    "shuffle=False"
                )
            CVClass = HoldOut
            cv = CVClass(
                test_size=test_size[ax],
                train_size=train_size[ax],
            )
        cv_list.append(copy.deepcopy(cv))

    return check_multipartite_cv(cv_list)


def check_multipartite_cv(
    cv,
    y=None,
    *,
    n_parts=2,  # Overwritten by cv if cv is list/tuple
    classifier=False,
    diagonal=False,
    shuffle=False,
    random_state=None,
):
    """
    Convert cross-validation objects into a `MultipartiteCrossValidator`.

    Parameters
    ----------
    cv : int, cross-validator, or tuple/list with such objects
        Determines the cross-validation splitting strategy. Possible inputs for
        cv are:

        - integer: is converted to a `KFold` instance with the provided number
          of folds.
        - A scikit-learn cross-validator instance.
        - A list or tuple of integers or cross-validation objects, specifying
          one of such objects for each axis of the multipartite dataset.

    y : array-like, default=None
        The interaction matrix (or tensor) that gathers the target values of
        a multipartite dataset.

    n_parts : int, default=2
        The number of parts in the multipartite dataset. E.g. 2 for a bipartite
        task.

    classifier : boolean, default=False
        Whether the task is a classification task. This parameter is deprecated
        and is not used in the function.

    diagonal : boolean, default=False
        Whether to perform diagonal cross-validation. See the :ref:`User Guide`
        for details.

    shuffle : boolean, default=False
        The `shuffle` parameter to be passed to KFold objects created from
        integers.

    random_state : int or RandomState instance, default=None
        The `random_state` parameter to be passed to KFold objects created from
        integers.

    Returns
    -------
    cv : MultipartiteCrossValidator
        A `MultipartiteCrossValidator` to split multipartite data.

    Notes
    -----
    The `shuffle` and `random_state` parameters are only used when `cv` is an
    integer or contains integers.
    """
    if isinstance(cv, MultipartiteCrossValidator):
        return cv

    # TODO: StratifiedKFold is not implemented for multipartite data.
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)

    if isinstance(cv, (tuple, list)):
        cv = [
            KFold(n_splits=i, shuffle=shuffle, random_state=random_state)
            if isinstance(i, int)
            else i
            for i in cv
        ]

    return MultipartiteCrossValidator(
        cv,
        diagonal=diagonal,
        n_parts=n_parts,  # Overritten by cv if cv is list/tuple
    )


def multipartite_train_test_split(
    *data,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
    input_type=BipartiteDataset,
):
    """Split datasets into random train and test subsets.

    Quick utility that wraps input validation and
    ``next(ShuffleSplit().split(X, y))`` and application to input data
    into a single call for splitting (and optionally subsampling) data in a
    oneliner.
    Read more in the :ref:`User Guide <cross_validation>`.

    The parameters 'test_size', 'train_size', 'shuffle' and 'stratify' can be
    optionally passed as a tuple or list of values intended for each respective
    axis of the ND data. Otherwise, the single value is equally considered for
    all axes.

    Parameters
    ----------
    *data : list-like of (InputDataND or (X, y) list-like pairs)
        N-dimensional datasets to be splitted.
    test_size : float or int or list-like of (float or int), default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
        One can optionally pass values for each axis in a list or tuple. If a
        single value is provided, it will be considered for all axes.
    train_size : float or int or list-like of (float or int), default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
        One can optionally pass values for each axis in a list or tuple. If a
        single value is provided, it will be considered for all axes.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    shuffle : bool or list-like of bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
        One can optionally pass values for each axis in a list or tuple. If a
        single value is provided, it will be considered for all axes.
    stratify : array-like or list-like of array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.
        Read more in the :ref:`User Guide <stratification>`.
        One can optionally pass arrays for each axis in a list or tuple. If a
        single array is provided, it will be considered for all axes.
    input_type : type, default=BipartiteDataset
        The input data format, with n_samples and n_features attributes.
        Read more in the :ref:`User Guide <input_types>` (TODO).

    Returns
    -------
    splitting : dict,
        Each value will be a list of the sampled datasets provided.
        Each key indentifies if train or test indices were used on each axis.
        E.g. for a 2D dataset, "LT" key means learned (training) rows and
        test columns. "TL" means test rows and training columns. "TT" means the
        data were selected with test rows and test columns as well.
    """
    if stratify is not None:
        raise NotImplementedError("Stratification not implemented.")
    n_sets = len(data)
    if n_sets == 0:
        raise ValueError("At least one item required as input")

    data = list(data)  # We use assignment.

    for i in range(len(data)):
        if not isinstance(data[i], input_type):
            data[i] = input_type(*data[i])

    n_parts = data[0].n_parts
    n_samples = data[0].n_samples

    # n_train, n_test = _validate_shuffle_split(
    #     n_samples=n_samples[ax],
    #     test_size=test_size[ax],
    #     train_size=train_size[ax],
    #     default_test_size=0.25
    # )

    cv = make_multipartite_train_test_splitter(
        n_parts=n_parts,
        test_size=test_size,
        train_size=train_size,
        shuffle=shuffle,
        stratified=stratify is not None,
        random_state=random_state,
    )
    train_test = next(cv.split(X=data[0].X, y=stratify))
    ttc_indices = combine_train_test_indices(train_test, include_train=True)

    split_data = {}
    for ttc_name, ttc_index in ttc_indices.items():
        if n_sets == 1:
            sample = data[0][ttc_index].Xy
        else:
            sample = [d[ttc_index].Xy for d in data]
        split_data[ttc_name] = sample

    return split_data
