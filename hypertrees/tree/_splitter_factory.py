import warnings
import copy
from numbers import Integral, Real
import numpy as np
from sklearn.tree._criterion import Criterion
from sklearn.tree._splitter import Splitter
from sklearn.utils._param_validation import validate_params, Interval
from ._nd_splitter import Splitter2D
from ._nd_criterion import MSE_Wrapper2D
from ._semisupervised_criterion import (
    SemisupervisedCriterion,
    SSCompositeCriterion,
    MSE_Wrapper2DSS,
)

# @validate_params(dict(
#     splitters=[list, tuple, Splitter],
#     criteria=[list, tuple, Criterion],
#     n_samples=[None, list, tuple, ],
#     max_features=None,
#     n_outputs=1,
#     min_samples_leaf=1,
#     min_weight_leaf=0.0,
#     ax_min_samples_leaf=1,
#     ax_min_weight_leaf=0.0,
#     random_state=None,
#     criterion_wrapper_class=MSE_Wrapper2D,
# ))


def make_2d_splitter(
    splitters,
    criteria,
    *,
    n_samples=None,
    max_features=None,
    n_outputs=1,
    min_samples_leaf=1,
    min_weight_leaf=0.0,
    ax_min_samples_leaf=1,
    ax_min_weight_leaf=0.0,
    random_state=None,
    criterion_wrapper_class=MSE_Wrapper2D,
):
    """Factory function of Splitter2D instances.

    Utility function to simplificate Splitter2D instantiation.
    Parameters may be set to a single value or a 2-valued
    tuple or list, to specify them for each axis.

    ax_min_samples_leaf represents [min_rows_leaf, min_cols_leaf]
    """
    if not isinstance(n_samples, (list, tuple)):
        n_samples = [n_samples, n_samples]
    if not isinstance(max_features, (list, tuple)):
        max_features = [max_features, max_features]
    if not isinstance(n_outputs, (list, tuple)):
        n_outputs = [n_outputs, n_outputs]
    if not isinstance(ax_min_samples_leaf, (list, tuple)):
        ax_min_samples_leaf = [ax_min_samples_leaf, ax_min_samples_leaf]
    if not isinstance(ax_min_weight_leaf, (list, tuple)):
        ax_min_weight_leaf = [ax_min_weight_leaf, ax_min_weight_leaf]
    if not isinstance(splitters, (list, tuple)):
        splitters = [copy.deepcopy(splitters), copy.deepcopy(splitters)]
    if not isinstance(criteria, (list, tuple)):
        criteria = [copy.deepcopy(criteria), copy.deepcopy(criteria)]
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    for ax in range(2):
        # Make criterion
        if isinstance(criteria[ax], type):
            if n_samples[ax] is None:
                raise ValueError(
                    f"n_samples[{ax}] must be provided if criteria"
                    f"[{ax}]={criteria[ax]} is a Criterion type.")

            criteria[ax] = criteria[ax](
                n_outputs=n_outputs[ax],
                n_samples=n_samples[ax],
            )

        elif not isinstance(criteria[ax], Criterion):
            raise TypeError

        # Make splitter
        if isinstance(splitters[ax], type):
            if max_features[ax] is None:
                raise ValueError(
                    f"max_features[{ax}] must be provided if splitters"
                    f"[{ax}]={splitters[ax]} is a Splitter type.")
            splitters[ax] = splitters[ax](
                criterion=criteria[ax],
                max_features=max_features[ax],
                min_samples_leaf=ax_min_samples_leaf[ax],
                min_weight_leaf=ax_min_weight_leaf[ax],
                random_state=random_state,
            )
        elif criteria[ax] is not None:
            warnings.warn("Since splitters[ax] is not a class, the provided"
                          " criteria[ax] is being ignored.")

    # Wrap criteria.
    criterion_wrapper = \
        criterion_wrapper_class(splitters[0], splitters[1])

    # Wrap splitters.
    return Splitter2D(
        splitter_rows=splitters[0],
        splitter_cols=splitters[1],
        criterion_wrapper=criterion_wrapper,
        min_samples_leaf=min_samples_leaf,
        min_weight_leaf=min_weight_leaf,
    )


def make_2dss_splitter(
    splitters,
    supervised_criteria,
    unsupervised_criteria,
    ss_criteria,
    supervision=0.5,
    n_features=None,
    n_samples=None,
    max_features=None,
    n_outputs=1,
    min_samples_leaf=1,
    min_weight_leaf=0.0,
    ax_min_samples_leaf=1,
    ax_min_weight_leaf=0.0,
    random_state=None,
    criterion_wrapper_class=MSE_Wrapper2DSS,
):
    """Factory function of Splitter2D instances with semisupervised criteria.

    Utility function to simplificate Splitter2D instantiation.
    Parameters may be set to a single value or a 2-valued
    tuple or list, to specify them for each axis.
    """
    if not isinstance(n_samples, (list, tuple)):
        n_samples = [n_samples, n_samples]
    if not isinstance(n_features, (list, tuple)):
        n_features = [n_features, n_features]
    if not isinstance(n_outputs, (list, tuple)):
        n_outputs = [n_outputs, n_outputs]

    if not isinstance(supervision, (list, tuple)):
        supervision = [supervision, supervision]
    if not isinstance(ss_criteria, (list, tuple)):
        ss_criteria = [copy.deepcopy(ss_criteria) for i in range(2)]
    if not isinstance(supervised_criteria, (list, tuple)):
        supervised_criteria = [copy.deepcopy(
            supervised_criteria) for i in range(2)]
    if not isinstance(unsupervised_criteria, (list, tuple)):
        unsupervised_criteria = \
            [copy.deepcopy(unsupervised_criteria) for i in range(2)]

    # Make semi-supervised criteria
    for ax in range(2):
        if ss_criteria[ax] is None:
            ss_criteria[ax] = SSCompositeCriterion
        elif isinstance(ss_criteria[ax], SemisupervisedCriterion):
            continue
        elif isinstance(ss_criteria[ax], type):
            if not issubclass(ss_criteria[ax], SSCompositeCriterion):
                raise ValueError
        else:
            raise ValueError

        if isinstance(supervised_criteria[ax], type):
            if not issubclass(supervised_criteria[ax], Criterion):
                raise ValueError

        if isinstance(unsupervised_criteria[ax], type):
            if not issubclass(unsupervised_criteria[ax], Criterion):
                raise ValueError

        ss_criteria[ax] = ss_criteria[ax](
            supervision=supervision[ax],
            supervised_criterion=supervised_criteria[ax],
            unsupervised_criterion=unsupervised_criteria[ax],
            n_outputs=n_outputs[ax],
            n_features=n_features[ax],
            n_samples=n_samples[ax],
        )

    return make_2d_splitter(
        splitters=splitters,
        criteria=ss_criteria,  # Main change.
        n_samples=n_samples,
        max_features=max_features,
        n_outputs=n_outputs,
        min_samples_leaf=min_samples_leaf,
        min_weight_leaf=min_weight_leaf,
        ax_min_samples_leaf=ax_min_samples_leaf,
        ax_min_weight_leaf=ax_min_weight_leaf,
        random_state=random_state,
        criterion_wrapper_class=criterion_wrapper_class,
    )


@validate_params(dict(
    supervision=[Interval(Real, 0.0, 1.0, closed="both")],
    n_outputs=[Interval(Integral, 1, None, closed="left"), None],
    n_features=[Interval(Integral, 1, None, closed="left"), None],
    n_samples=[Interval(Integral, 1, None, closed="left"), None],
    ss_class=[type],
    supervised_criterion=[Criterion, type],
    unsupervised_criterion=[Criterion, type],
))
def make_semisupervised_criterion(
    supervision,
    ss_class,
    n_outputs=None,
    n_features=None,
    n_samples=None,
    supervised_criterion=None,
    unsupervised_criterion=None,
):
    if isinstance(supervised_criterion, type):
        if not n_outputs or not n_samples:
            raise ValueError('If supervised_criterion is a class, one must'
                             ' provide both n_outputs (received '
                             f'{n_outputs}) and n_samples ({n_samples}).')
        supervised_criterion = supervised_criterion(
            n_outputs=n_outputs,
            n_samples=n_samples,
        )
    if isinstance(unsupervised_criterion, type):
        if not n_features or not n_samples:
            raise ValueError('If unsupervised_criterion is a class, one mu'
                             'st provide both n_features (received '
                             f'{n_features}) and n_samples ({n_samples}).')
        unsupervised_criterion = unsupervised_criterion(
            n_outputs=n_features,
            n_samples=n_samples,
        )

    return ss_class(
        supervision=supervision,
        supervised_criterion=supervised_criterion,
        unsupervised_criterion=unsupervised_criterion,
    )
