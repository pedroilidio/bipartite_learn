from abc import ABCMeta, abstractmethod
from typing import Callable, Dict
import numpy as np
import pytest
import scipy.stats

from collections import defaultdict
from numbers import Real, Integral
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_random_state
from sklearn.utils._param_validation import validate_params, Interval
from bipartite_learn.tree._splitter_factory import (
    make_semisupervised_criterion,
    make_bipartite_criterion,
    make_2dss_criterion,
)
from bipartite_learn.tree._axis_criterion import (
    AxisSquaredError,
    AxisSquaredErrorGSO,
    AxisFriedmanGSO,
    AxisGini,
    AxisEntropy,
)
from bipartite_learn.tree._unsupervised_criterion import (
    PairwiseFriedman,
    PairwiseSquaredError,
    PairwiseSquaredErrorGSO,
    UnsupervisedSquaredError,
    UnsupervisedGini,
    UnsupervisedEntropy,
    # UnsupervisedSquaredErrorGSO,  # TODO
    UnsupervisedFriedman,
    MeanDistance,
)
from bipartite_learn.tree._bipartite_criterion import GMO, GMOSA
from bipartite_learn.tree._semisupervised_criterion import (
    SSCompositeCriterion,
)

from bipartite_learn.melter import row_cartesian_product
from .utils.test_utils import assert_equal_dicts, comparison_text
from .utils.make_examples import make_interaction_blobs
from .utils.tree_utils import (
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

PAIRWISE_CRITERIA = {
    MeanDistance,
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
# Criterion objects to compare against
# =============================================================================


class BaseReferenceCriterion(metaclass=ABCMeta):
    def __init__(self, average_both_axes=False):
        self.average_both_axes = average_both_axes

    def set_data(self, X, y, sample_weight=None, **kwargs):
        self.X = X
        self.y = y
        self.sample_weight = sample_weight

    @abstractmethod
    def impurity(self, y):
        ...

    def impurity_improvement(self, split_data: dict):
        return (
            split_data['weighted_n_node_samples'] /
            split_data['weighted_n_samples']
            * (
                split_data['impurity_parent']

                - split_data['impurity_left']
                * split_data['weighted_n_left']
                / split_data['weighted_n_node_samples']

                - split_data['impurity_right']
                * split_data['weighted_n_right']
                / split_data['weighted_n_node_samples']
            )
        )
    
    def children_impurity(self, y_region, rel_pos):
        return (
            self.impurity(y_region[:rel_pos]),
            self.impurity(y_region[rel_pos:]),
        )

    @validate_params({
        'start': [Integral],
        'end': [Integral],
        'feature': [Interval(Integral, 0, None, closed='left'), None],
    })
    def evaluate_split(
        self,
        pos: int,
        start: int = 0,
        end: int = 0,
        *,
        feature: int | None = None,
        weighted_n_samples: float | None = None,
        weighted_n_node_samples: float | None = None,
    ) -> Dict[str, int | float]:
        """Evaluate impurities and improvements of a split position.

        Calculate the impurities and impurity improvement of a given split
        position in a dataset, simulating split evaluation by the Criterion
        objects of decision trees.

        Parameters
        ----------
        pos : int, optional (default=0)
            The split position to evaluate. The absolute index must be provided,
            relative to the beginning of the dataset, NOT relative to `start`.
        start : int, optional (default=0)
            The starting position of the data to evaluate, representing the start
            index of the current node, by default 0.
        end : int, optional (default=0)
            The ending position of the data to evaluate, representing the final
            index of the current node, by default 0.
        feature : int, optional (default=None)
            The feature index representing the column of X to sort the data by. If
            None, no sorting is performed. By default None.
        weighted_n_samples : float, optional (default=None)
            The total weighted number of samples in the dataset, taken as
            `sample_weight.sum()` if None.
        weighted_n_node_samples : float, optional (default=None)
            The total weighted number of samples in the current node, taken to be
            `sample_weight[start:end].sum()` if None.

        Returns
        -------
        Dict[str, Number]
            A dictionary with at least the following keys and values:
            - 'pos': the split position.
            - 'impurity_parent': the impurity of the parent (current) node.
            - 'weighted_n_left': the weighted number of samples in the left node.
            - 'weighted_n_right': the weighted number of samples in the right node.
            - 'impurity_left': the impurity of the left node.
            - 'impurity_right': the impurity of the right node.
            - 'improvement': the impurity improvement gained by the split.
        """
        start = start if start >= 0 else self.y.shape[0] + start
        end = end if end > 0 else self.y.shape[0] + end

        if not (start < pos <= end):
            raise ValueError(
                f'Provided index {pos} is not between {start=} and {end=}.'
            )

        x_ = self.X[start:end].copy()
        y_ = self.y[start:end].copy()
        rel_pos = pos - start

        if feature is not None:
            x_, y_ = sort_by_feature(x_, y_, feature)

        n_samples = self.y.shape[0]
        n_node_samples = y_.shape[0]

        result = {
            'pos': pos,
            'impurity_parent': self.impurity(y_),
            'start': start,
            'end': end,
        }

        if self.sample_weight is None:
            result['weighted_n_left'] = float(rel_pos)
            weighted_n_samples = float(n_samples)
            weighted_n_node_samples = float(n_node_samples)
        else:
            result['weighted_n_left'] = self.sample_weight[start:pos].sum()
            if weighted_n_samples is None:
                weighted_n_samples = self.sample_weight.sum()
            if weighted_n_node_samples is None:
                weighted_n_node_samples = self.sample_weight[start:end].sum()

        result['weighted_n_right'] = (
            weighted_n_node_samples - result['weighted_n_left']
        )

        result['n_samples'] = n_samples
        result['n_node_samples'] = n_node_samples
        result['weighted_n_samples'] = weighted_n_samples
        result['weighted_n_node_samples'] = weighted_n_node_samples

        result['impurity_left'], result['impurity_right'] = (
            self.children_impurity(y_, rel_pos)
        )

        if self.average_both_axes:
            result['impurity_parent'] += self.impurity(y_.T)
            result['impurity_parent'] /= 2
            result['impurity_left'] += self.impurity(y_[:rel_pos].T)
            result['impurity_left'] /= 2
            result['impurity_right'] += self.impurity(y_[rel_pos:].T)
            result['impurity_right'] /= 2

        result['improvement'] = self.impurity_improvement(result)

        return result

    def eval_all_splits(
        self,
        start=0,
        end=0,
        indices=None,
    ):
        """Evaluate splits for all positions in a given range of indices.

        Parameters
        ----------
        start : int, optional (default=0)
            The starting index for the range of indices to evaluate splits for.
            If negative, it is considered as counting from the end of the
            samples.

        end : int, optional (default=0)
            The ending index for the range of indices to evaluate splits for.
            If negative, it is considered as counting from the end of the
            samples.

        indices : iterable of int, optional (default=None)
            The indices to evaluate splits for. If None, the range of indices
            from `start + 1` to `end` will be used.

        Returns
        -------
        dict[str, Number | np.ndarray]
            A dictionary containing the result of evaluating splits for all
            positions in the specified range of indices. The keys of the
            dictionary are the names of the criterion measures, and the values
            are scalars for node-wise info or arrays for results in each
            position.
        """
        start = start if start >= 0 else self.y.shape[0] + start
        end = end if end > 0 else self.y.shape[0] + end

        if self.sample_weight is None:
            weighted_n_samples = float(self.y.shape[0])
            weighted_n_node_samples = float(end - start)
        else:
            weighted_n_samples = self.sample_weight.sum()
            weighted_n_node_samples = self.sample_weight[start:end].sum()

        if indices is None:
            indices = range(start + 1, end)

        result = defaultdict(list)

        for pos in indices:
            split = self.evaluate_split(
                pos=pos,
                start=start,
                end=end,
                weighted_n_samples=weighted_n_samples,
                weighted_n_node_samples=weighted_n_node_samples,
            )
            for k, v in split.items():
                result[k].append(v)

        result = {k: np.array(v) for k, v in result.items()}
        result['weighted_n_samples'] = weighted_n_samples
        result['weighted_n_node_samples'] = weighted_n_node_samples

        return result


class ReferenceGini(BaseReferenceCriterion):
    def impurity(self, y):
        sq_sums = 0
        for col in y.T:
            _, counts = np.unique(col, return_counts=True)
            sq_sums += np.square(counts / y.shape[0]).sum()
        return 1 - sq_sums / y.shape[1]


class ReferenceEntropy(BaseReferenceCriterion):
    def impurity(self, y):
        col_entropies = 0
        for col in y.T:
            _, counts = np.unique(col, return_counts=True)
            p = counts / y.shape[0]
            col_entropies -= (p * np.log2(p)).sum()
        return col_entropies / y.shape[1]


class ReferenceSquaredError(BaseReferenceCriterion):
    def impurity(self, y):
        return y.var(0).mean()


class ReferenceSquaredErrorGSO(BaseReferenceCriterion):
    def impurity(self, y):
        return y.var()


class ReferenceFriedman(ReferenceSquaredError):
    def impurity_improvement(self, split_data):
        total_sum_left = self.y[:split_data['pos']].sum()
        total_sum_right = self.y[split_data['pos']:].sum()

        diff = (
            split_data['weighted_n_right'] * total_sum_left
            - split_data['weighted_n_left'] * total_sum_right
        )

        return diff ** 2 / (
            split_data['weighted_n_right']
            * split_data['weighted_n_left']
            * split_data['weighted_n_node_samples']
            * self.y.shape[1] ** 2
        )


class ReferenceFriedmanGSO(ReferenceSquaredErrorGSO):
    def impurity_improvement(self, split_data):
        total_sum_left = self.y[split_data['start']:split_data['pos']].sum()
        total_sum_right = self.y[split_data['pos']:split_data['end']].sum()

        diff = (
            split_data['weighted_n_right'] * total_sum_left
            - split_data['weighted_n_left'] * total_sum_right
        )

        return diff ** 2 / (
            split_data['weighted_n_right']
            * split_data['weighted_n_left']
            * split_data['weighted_n_node_samples']
            * self.y.shape[1]  # weighted_n_node_cols
        )


class ReferenceMeanDistance(BaseReferenceCriterion):
    def impurity(self, y):
        if y.shape[0] != y.shape[1]:
            raise ValueError(f'y must be square. Received {y.shape=}')
        if y.shape[0] <= 1:
            return 0.0
        return y[np.triu_indices(y.shape[0], 1)].mean()

    def children_impurity(self, y_region, rel_pos):
        y_left = y_region[:rel_pos, :rel_pos]
        y_right = y_region[rel_pos:, rel_pos:]
        return (
            self.impurity(y_left),
            self.impurity(y_right),
        )
    
    def evaluate_split(self, *args, **kwargs):
        split = super().evaluate_split(*args, **kwargs)

        if self.sample_weight is None:
            split['weighted_n_left'] = (
                (split['weighted_n_left'] - 1)
                * split['weighted_n_left'] / 2
            )
            split['weighted_n_right'] = (
                (split['weighted_n_right'] - 1)
                * split['weighted_n_right'] / 2
            )
            split['weighted_n_samples'] = (
                (split['weighted_n_samples'] - 1)
                * split['weighted_n_samples'] / 2
            )
            split['weighted_n_node_samples'] = (
                (split['weighted_n_node_samples'] - 1)
                * split['weighted_n_node_samples'] / 2
            )
        else:
            raise NotImplementedError
        
        return split


class ReferenceCompositeSS(BaseReferenceCriterion):
    def __init__(
        self,
        sup_criterion,
        unsup_criterion,
        average_both_axes=False,
    ):
        self.sup_criterion = sup_criterion
        self.unsup_criterion = unsup_criterion
        self.average_both_axes = average_both_axes
        self.sup_criterion.average_both_axes = average_both_axes

    def set_data(self, X, y, supervision, sample_weight=None):
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.supervision = supervision
        self.sup_criterion.set_data(X, y, sample_weight)
        self.unsup_criterion.set_data(X, X, sample_weight)

    def evaluate_split(
        self,
        *args,
        **kwargs,
    ) -> Dict[str, int | float]:

        sup_split = self.sup_criterion.evaluate_split(*args, **kwargs)
        unsup_split = self.unsup_criterion.evaluate_split(*args, **kwargs)

        # sup_split['root_impurity'] = sup_split['impurity_parent']
        # unsup_split['root_impurity'] = unsup_split['impurity_parent']

        unsup_split['improvement'] = \
            self.unsup_criterion.impurity_improvement(unsup_split)
        sup_split['improvement'] = \
            self.sup_criterion.impurity_improvement(sup_split)

        result = {
            **{'supervised_' + k: v for k, v in sup_split.items()},
            **{'unsupervised_' + k: v for k, v in unsup_split.items()},
        }

        result['impurity_parent'] = 1.0

        result['impurity_left'] = self.combine_impurities(
            sup_split['impurity_left'] / sup_split['impurity_parent'],
            unsup_split['impurity_left'] / unsup_split['impurity_parent'],
        )
        result['impurity_right'] = self.combine_impurities(
            sup_split['impurity_right'] / sup_split['impurity_parent'],
            unsup_split['impurity_right'] / unsup_split['impurity_parent'],
        )

        result['improvement'] = self.impurity_improvement(result)
        result['pos'] = sup_split['pos']

        return result

    def combine_impurities(self, sup_impurity, unsup_impurity):
        if self.average_both_axes:
            # unsupervised impurity of the other axis is 1.0, since it does
            # not change.
            #  imp = imp + imp_other_axis
            unsup_impurity += 1
            unsup_impurity /= 2

        return (
            self.supervision * sup_impurity
            + (1 - self.supervision) * unsup_impurity
        )

    def impurity_improvement(self, split_data: dict):
        sup_improvement = (
            self.supervision
            * split_data['supervised_improvement']
            / split_data['supervised_impurity_parent']
        )
        unsup_improvement = (
            + (1 - self.supervision)
            * split_data['unsupervised_improvement']
            / split_data['unsupervised_impurity_parent']
        )
        if self.average_both_axes:  # The improvement on the other axis is zero
            unsup_improvement /= 2

        return sup_improvement + unsup_improvement

    def impurity(self):
        return None


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
            ReferenceCompositeSS(ReferenceGini(), ReferenceSquaredError()),
        ),
        (
            make_semisupervised_criterion,
            {
                'supervised_criterion': UnsupervisedEntropy,
                'unsupervised_criterion': UnsupervisedSquaredError,
                'ss_class': SSCompositeCriterion,
            },
            ReferenceCompositeSS(ReferenceEntropy(), ReferenceSquaredError()),
        ),
        (
            make_semisupervised_criterion,
            {
                'supervised_criterion': UnsupervisedSquaredError,
                'unsupervised_criterion': UnsupervisedSquaredError,
            },
            ReferenceCompositeSS(
                ReferenceSquaredError(),
                ReferenceSquaredError(),
            ),
        ),
        (
            make_semisupervised_criterion,
            {
                'supervised_criterion': UnsupervisedFriedman,
                'unsupervised_criterion': UnsupervisedFriedman,
            },
            ReferenceCompositeSS(
                ReferenceFriedman(),
                ReferenceFriedman(),
            ),
        ),
        (
            make_semisupervised_criterion,
            {
                'supervised_criterion': UnsupervisedFriedman,
                'unsupervised_criterion': UnsupervisedSquaredError,
            },
            ReferenceCompositeSS(
                ReferenceFriedman(),
                ReferenceSquaredError(),
            ),
        ),
        (
            make_semisupervised_criterion,
            {
                'supervised_criterion': UnsupervisedSquaredError,
                'unsupervised_criterion': MeanDistance,
            },
            ReferenceCompositeSS(
                ReferenceSquaredError(),
                ReferenceMeanDistance(),
            ),
        ),
    ],
    ids=[
        'gini|mse',
        'entropy|mse',
        'mse',
        'friedman',
        'friedman|mse',
        'mse|mean_distance',
    ],
)
def semisupervised_criterion(
    request,
    n_samples,
    n_features,
    n_classes,
    supervision,
):
    factory, args, ref_criterion = request.param

    default_criterion_args = dict(
        n_features=n_features[0],
        n_samples=n_samples[0],
        n_classes=n_classes[0],
        n_outputs=n_samples[1],
        supervision=supervision,
    )

    return {
        'criterion': factory(**default_criterion_args | args),
        'ref_criterion': ref_criterion,
    }


@pytest.fixture(
    params=[
        (
            make_2dss_criterion,
            {
                'supervised_criteria': AxisGini,
                'unsupervised_criteria': UnsupervisedSquaredError,
                'ss_criteria': SSCompositeCriterion,
                'bipartite_criterion_class': GMOSA,
            },
            ReferenceCompositeSS(
                ReferenceGini(),
                ReferenceSquaredError(),
                average_both_axes=True,
            ),
        ),
        (
            make_2dss_criterion,
            {
                'supervised_criteria': AxisEntropy,
                'unsupervised_criteria': UnsupervisedSquaredError,
                'ss_criteria': SSCompositeCriterion,
                'bipartite_criterion_class': GMOSA,
            },
            ReferenceCompositeSS(
                ReferenceEntropy(),
                ReferenceSquaredError(),
                average_both_axes=True,
            ),
        ),
        (
            make_2dss_criterion,
            {
                'supervised_criteria': AxisSquaredErrorGSO,
                'unsupervised_criteria': UnsupervisedSquaredError,
                'ss_criteria': SSCompositeCriterion,
                'bipartite_criterion_class': GMOSA,
            },
            ReferenceCompositeSS(
                ReferenceSquaredErrorGSO(),
                ReferenceSquaredError(),
                average_both_axes=True,
            ),
        ),
        (
            make_2dss_criterion,
            {
                'supervised_criteria': AxisSquaredError,
                'unsupervised_criteria': UnsupervisedSquaredError,
                'ss_criteria': SSCompositeCriterion,
                'bipartite_criterion_class': GMOSA,
            },
            ReferenceCompositeSS(
                ReferenceSquaredError(),
                ReferenceSquaredError(),
                average_both_axes=True,
            ),
        ),
        (
            make_2dss_criterion,
            {
                'supervised_criteria': AxisFriedmanGSO,
                'unsupervised_criteria': UnsupervisedSquaredError,
                'ss_criteria': SSCompositeCriterion,
                'bipartite_criterion_class': GMOSA,
            },
            ReferenceCompositeSS(
                ReferenceFriedmanGSO(),
                ReferenceSquaredError(),
                average_both_axes=True,
            ),
        ),
        (
            make_2dss_criterion,
            {
                'supervised_criteria': AxisSquaredErrorGSO,
                'unsupervised_criteria': MeanDistance,
                'ss_criteria': SSCompositeCriterion,
                'bipartite_criterion_class': GMOSA,
            },
            ReferenceCompositeSS(
                ReferenceSquaredErrorGSO(),
                ReferenceMeanDistance(),
                average_both_axes=True,
            ),
        ),
    ],
    ids=[
        'gini|mse',
        'entropy|mse',
        'mse_gso',
        'mse_gmo',
        'friedman|mse',
        'mse_gso|mean_distance',
    ],
)
def ss_bipartite_criterion(
    request,
    n_samples,
    n_features,
    supervision,
    n_classes,
):
    factory, args, ref_criterion = request.param

    default_criterion_args = dict(
        n_features=n_features,
        n_samples=n_samples,
        n_classes=n_classes,
        n_outputs=n_samples[::-1],
        supervision=supervision,
    )

    return {
        'criterion': factory(**default_criterion_args | args),
        'ref_criterion': ref_criterion,
    }


@pytest.fixture(
    params=[
        (
            make_bipartite_criterion,
            {
                'criteria': AxisGini,
                'bipartite_criterion_class': GMOSA,
            },
            ReferenceGini(average_both_axes=True),
        ),
        (
            make_bipartite_criterion,
            {
                'criteria': AxisEntropy,
                'bipartite_criterion_class': GMOSA,
            },
            ReferenceEntropy(average_both_axes=True),
        ),
        (
            make_bipartite_criterion,
            {
                'criteria': AxisSquaredErrorGSO,
                'bipartite_criterion_class': GMOSA,
            },
            ReferenceSquaredErrorGSO(),
        ),
        (
            make_bipartite_criterion,
            {
                'criteria': AxisSquaredError,
                'bipartite_criterion_class': GMOSA,
            },
            ReferenceSquaredError(average_both_axes=True),
        ),
        (
            make_bipartite_criterion,
            {
                'criteria': AxisFriedmanGSO,
                'bipartite_criterion_class': GMOSA,
            },
            ReferenceFriedmanGSO(average_both_axes=True),
        ),
    ],
    ids=['gini', 'entropy', 'mse_gso', 'mse_gmo', 'friedman'],
)
def supervised_bipartite_criterion(request, n_samples, n_classes):
    factory, args, ref_criterion = request.param

    default_criterion_args = dict(
        n_samples=n_samples,
        n_outputs=n_samples[::-1],
        n_classes=n_classes,
    )
    return {
        'criterion': factory(**default_criterion_args | args),
        'ref_criterion': ref_criterion,
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


def turn_into_pairwise(x):
    return rbf_kernel(x)


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
    ref_criterion = ss_criterion_data['ref_criterion']

    if (
        type(criterion) in CLASSIFICATION_CRITERIA
        or supervision is not None
        and type(criterion.supervised_criterion) in CLASSIFICATION_CRITERIA
    ):
        Y = turn_into_classification(Y)

    if (
        supervision is not None
        and type(criterion.unsupervised_criterion) in PAIRWISE_CRITERIA
    ):
        X = turn_into_pairwise(X)

    criterion.set_X(X)
    split_data = apply_criterion(criterion, Y)

    ref_criterion.set_data(X, Y, supervision=supervision)
    ref_split_data = ref_criterion.eval_all_splits()

    return {'splits': split_data, 'reference_splits': ref_split_data}


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
    ref_criterion = ss_bipartite_criterion['ref_criterion']

    if (
        type(criterion.criterion_rows) in CLASSIFICATION_CRITERIA
        or supervision is not None
        and type(criterion.criterion_rows.supervised_criterion)
        in CLASSIFICATION_CRITERIA
    ):
        Y = turn_into_classification(Y)

    if supervision is not None:
        if (
            type(criterion.criterion_rows.unsupervised_criterion)
            in PAIRWISE_CRITERIA
        ):
            X[0] = turn_into_pairwise(X[0])
        if (
            type(criterion.criterion_cols.unsupervised_criterion)
            in PAIRWISE_CRITERIA
        ):
            X[1] = turn_into_pairwise(X[1])

    row_splits, col_splits = apply_criterion(
        criterion,
        X[0],
        X[1],
        Y,
        start=[start_row, start_col],
        end=[end_row, end_col],
    )

    ref_criterion.set_data(X[0], Y, supervision=supervision)
    ref_row_splits = ref_criterion.eval_all_splits(
        start=start_row, end=end_row,
    )

    ref_criterion.set_data(X[1], Y.T, supervision=supervision)
    ref_col_splits = ref_criterion.eval_all_splits(
        start=start_col, end=end_col,
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

    ref_splits = semisupervised_splits['reference_splits']
    ref_splits['ss_improvement'] = ref_splits['improvement']
    ref_splits['ss_impurity_parent'] = ref_splits['impurity_parent']
    ref_splits['ss_impurity_right'] = ref_splits['impurity_right']
    ref_splits['ss_impurity_left'] = ref_splits['impurity_left']

    assert_equal_dicts(
        semisupervised_splits['splits'],
        ref_splits,
        rtol=rtol,
        atol=atol,
        ignore={
            'n_samples',
            'n_node_samples',
            'proxy_improvement',
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

    ref_criterion = ReferenceSquaredError()
    ref_criterion.set_data(X, Y)
    ref_split_data = ref_criterion.eval_all_splits()

    assert_equal_dicts(
        split_data,
        ref_split_data,
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
        ref_split_data['improvement'],
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

    ignore = {
        'n_samples',
        'n_node_samples',
        'supervised_n_samples',
        'supervised_n_node_samples',
        'unsupervised_n_samples',
        'unsupervised_n_node_samples',
        'weighted_n_samples',
        'weighted_n_node_samples',
    }

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
    'single_output_ref_criterion, multioutput_ref_criterion', [
        (ReferenceSquaredErrorGSO(), ReferenceSquaredError()),
        (ReferenceFriedmanGSO(), ReferenceFriedman()),
        (
            ReferenceCompositeSS(
                ReferenceSquaredErrorGSO(),
                ReferenceSquaredError(),
            ),
            ReferenceCompositeSS(
                ReferenceSquaredError(),
                ReferenceSquaredError(),
            ),
        )
    ],
    ids=['mse', 'friedman', 'ss_mse'],
)
def test_gso_gmo_equivalence(
    data,
    n_samples,
    supervision,
    single_output_ref_criterion,
    multioutput_ref_criterion,
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

    multioutput_ref_criterion.set_data(
        X=np.repeat(X[0], n_cols, axis=0),
        y=Y.reshape(-1, 1),
        supervision=supervision,
    )
    mo_row_splits = multioutput_ref_criterion.eval_all_splits(
        indices=row_indices * n_cols,
        start=start_row * n_cols,
        end=end_row * n_cols,
    )

    multioutput_ref_criterion.set_data(
        X=np.repeat(X[1], n_rows, axis=0),
        y=Y.T.reshape(-1, 1),
        supervision=supervision,
    )
    mo_col_splits = multioutput_ref_criterion.eval_all_splits(
        indices=col_indices * n_rows,
        start=start_col * n_rows,
        end=end_col * n_rows,
    )

    single_output_ref_criterion.set_data(X[0], Y, supervision=supervision)
    so_row_splits = single_output_ref_criterion.eval_all_splits(
        start=start_row,
        end=end_row,
    )

    single_output_ref_criterion.set_data(X[1], Y.T, supervision=supervision)
    so_col_splits = single_output_ref_criterion.eval_all_splits(
        start=start_col,
        end=end_col,
    )

    mo_row_splits['pos'] = row_indices
    mo_col_splits['pos'] = col_indices

    ignore = {
        'n_samples',
        'n_node_samples',
        'weighted_n_left',
        'weighted_n_right',
        'weighted_n_samples',
        'weighted_n_node_samples',
        'supervised_n_samples',
        'supervised_n_node_samples',
        'supervised_weighted_n_left',
        'supervised_weighted_n_right',
        'supervised_weighted_n_samples',
        'supervised_weighted_n_node_samples',
        'unsupervised_n_samples',
        'unsupervised_n_node_samples',
        'unsupervised_weighted_n_left',
        'unsupervised_weighted_n_right',
        'unsupervised_weighted_n_samples',
        'unsupervised_weighted_n_node_samples',
        'supervised_pos',
        'unsupervised_pos',
        'start',
        'end',
        'supervised_start',
        'supervised_end',
        'unsupervised_start',
        'unsupervised_end',
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
