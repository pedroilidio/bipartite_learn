import itertools
import copy
import numpy as np
from sklearn.model_selection._split import check_cv, BaseCrossValidator

def check_cv_nd(cv, y, classifier, diagonal):
    ndim = y.ndim  # FIXME: not valid for multi-output.

    if type(cv) not in (tuple, list):
        cv = [copy.deepcopy(cv) for _ in range(ndim)]

    return CrossValidatorNDWrapper(
        # FIXME: transposing y may be necessary.
        [check_cv(cvi, y, classifier=classifier) for cvi in cv],
        diagonal=diagonal,
    )


class CrossValidatorNDWrapper(BaseCrossValidator):
    def __init__(self, cross_validators, diagonal):
        self.cross_validators = cross_validators
        self.diagonal = diagonal
        self.ndim = len(cross_validators)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._check_n_splits(X, y, groups)

    def _check_n_splits(self, X=None, y=None, groups=None):
        ax_n_splits = self._get_from_children_cv(
            'get_n_splits', X, y, groups)

        if self.diagonal:
            # Ensure all cross validators report the same number of splits.
            g = itertools.groupby(ax_n_splits)
            next(g, None)

            # if not all elements in ax_n_splits are equal:
            if next(g, False):
                raise ValueError("Cross-validators must generate the same number"
                                 " of splits if diagonal=True")
            return next(ax_n_splits)

        return np.prod(list(ax_n_splits))

    def split(self, X, y=None, groups=None):
        train_test = self._get_from_children_cv('split', X, y=None, groups=None)
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

        for split in split_iter:
            yield split

    def _iter_test_masks(self, X=None, y=None, groups=None):
        iter_ = self._get_from_children_cv(
            '_iter_test_masks', X, y=None, groups=None)

        for item in iter_:
            yield item

    def _iter_test_indices(self, X=None, y=None, groups=None):
        iter_ = self._get_from_children_cv(
            '_iter_test_indices', X, y=None, groups=None)

        for item in iter_:
            yield item

    def _get_from_children_cv(self, attr, X=None, y=None, groups=None):
        if X is None:
            X = [None] * self.ndim
        if groups is None:
            groups = [None] * self.ndim
        if y is None:
            y = [None] * self.ndim
        else:
            # FIXME: is np.moveaxis well suited here?
            y = [np.moveaxis(y, ax, 0) for ax in range(self.ndim)]

        for cv, Xi, yi, group in zip(self.cross_validators, X, y, groups):
            yield getattr(cv, attr)(Xi, yi, group)
