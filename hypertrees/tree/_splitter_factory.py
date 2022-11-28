import warnings
import copy
from numbers import Integral, Real
import numpy as np
from sklearn.tree._criterion import Criterion
from sklearn.tree._splitter import Splitter
from sklearn.utils.validation import check_random_state
from sklearn.utils._param_validation import (
    validate_params, Interval, HasMethods
)
from ._nd_splitter import Splitter2D
from ._nd_criterion import MSE_Wrapper2D
from ._semisupervised_criterion import (
    SemisupervisedCriterion,
    SSCompositeCriterion,
    BipartiteSemisupervisedCriterion,
)


CriterionConstraints = [
]


def _duplicate_single_parameters(*params, ndim=2):
    result = []

    for param in params:
        if isinstance(param, (list, tuple)):
            if len(param) != ndim:
                raise ValueError(
                    f"param {param=} has length {len(param)}, expected {ndim}."
                )
            result.append(param)
        else:
            result.append([copy.deepcopy(param) for _ in range(ndim)])

    return result


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
    (
        n_samples,
        max_features,
        n_outputs,
        ax_min_samples_leaf,
        ax_min_weight_leaf,
        splitters,
        criteria,
    ) = _duplicate_single_parameters(
        n_samples,
        max_features,
        n_outputs,
        ax_min_samples_leaf,
        ax_min_weight_leaf,
        splitters,
        criteria,
    )

    random_state = check_random_state(random_state)

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
        criterion_wrapper_class(criteria[0], criteria[1])

    # Wrap splitters.
    return Splitter2D(
        splitter_rows=splitters[0],
        splitter_cols=splitters[1],
        criterion_wrapper=criterion_wrapper,
        min_samples_leaf=min_samples_leaf,
        min_weight_leaf=min_weight_leaf,
    )


# cdef class SSCompositeCriterionAlves(SSCompositeCriterion):
#     """Unsupervised impurity is only used to decide between rows or columns.
# 
#     The split search takes into consideration only the labels, as usual, but
#     after the rows splitter and the columns splitter defines each one's split,
#     unsupervised information is used to decide between them, i.e. the final
#     impurity is semisupervised as in MSE_wrapper2DSS, but the proxy improvement
#     only uses supervised data.
#     """
def make_2dss_splitter(
    splitters,
    supervised_criteria=None,  # FIXME: validate, cannot pass None
    unsupervised_criteria=None,
    ss_criteria=None,
    supervision=0.5,
    update_supervision=None,
    n_features=None,
    n_samples=None,
    max_features=None,
    n_outputs=1,
    min_samples_leaf=1,
    min_weight_leaf=0.0,
    ax_min_samples_leaf=1,
    ax_min_weight_leaf=0.0,
    random_state=None,
    pairwise=False,
    axis_decision_only=False,
    criterion_wrapper_class=MSE_Wrapper2D,
    ss_criterion_wrapper_class=BipartiteSemisupervisedCriterion,
):
    """Factory function of Splitter2D instances with semisupervised criteria.

    Utility function to simplificate Splitter2D instantiation.
    Parameters may be set to a single value or a 2-valued
    tuple or list, to specify them for each axis.
    """
    (
        n_samples,
        n_features,
        max_features,
        n_outputs,
        supervision,
        ax_min_samples_leaf,
        ax_min_weight_leaf,
        splitters,
        ss_criteria,
        supervised_criteria,
        unsupervised_criteria,
    ) = _duplicate_single_parameters(
        n_samples,
        n_features,
        max_features,
        n_outputs,
        supervision,
        ax_min_samples_leaf,
        ax_min_weight_leaf,
        splitters,
        ss_criteria,
        supervised_criteria,
        unsupervised_criteria,
    )

    random_state = check_random_state(random_state)

    # Make semi-supervised criteria
    # TODO: warn about skipped arguments
    for ax in range(2):
        if isinstance(splitters[ax], Splitter):
            continue
        elif not issubclass(splitters[ax], Splitter):  # TODO: use duck typing?
            raise ValueError(
                "splitter must be instance or subclass of "
                "sklearn.tree._splitter.Splitter"
            )

        if ss_criteria[ax] is None:
            ss_criteria[ax] = SSCompositeCriterion

        if not isinstance(ss_criteria[ax], SemisupervisedCriterion):
            ss_criteria[ax] = make_semisupervised_criterion(
                ss_class=ss_criteria[ax],
                supervision=supervision[ax],
                supervised_criterion=supervised_criteria[ax],
                unsupervised_criterion=unsupervised_criteria[ax],
                n_outputs=n_outputs[ax],
                n_features=n_features[ax],
                n_samples=n_samples[ax],
            )

        splitters[ax] = splitters[ax](
            criterion=ss_criteria[ax],
            max_features=max_features[ax],
            min_samples_leaf=ax_min_samples_leaf[ax],
            min_weight_leaf=ax_min_weight_leaf[ax],
            random_state=random_state,
        )
    
    criterion_wrapper = ss_criterion_wrapper_class(
        unsupervised_criterion_rows=ss_criteria[0].unsupervised_criterion,
        unsupervised_criterion_cols=ss_criteria[1].unsupervised_criterion,
        supervised_bipartite_criterion=criterion_wrapper_class(
            criterion_rows=ss_criteria[0].supervised_criterion,
            criterion_cols=ss_criteria[1].supervised_criterion,
        ),
        supervision_rows=supervision[0],
        supervision_cols=supervision[1],
        update_supervision=update_supervision,
    )

    return Splitter2D(
        splitter_rows=splitters[0],
        splitter_cols=splitters[1],
        criterion_wrapper=criterion_wrapper,
        min_samples_leaf=min_samples_leaf,
        min_weight_leaf=min_weight_leaf,
    )


@validate_params(dict(
    supervision=[Interval(Real, 0.0, 1.0, closed="both")],
    n_outputs=[Interval(Integral, 1, None, closed="left"), None],
    n_features=[Interval(Integral, 1, None, closed="left"), None],
    n_samples=[Interval(Integral, 1, None, closed="left"), None],
    ss_class=[type(SemisupervisedCriterion)],
    supervised_criterion=[Criterion, type(Criterion)],
    unsupervised_criterion=[Criterion, type(Criterion)],
    update_supervision=[callable, None],
))
def make_semisupervised_criterion(
    supervision,
    ss_class,
    n_outputs=None,
    n_features=None,
    n_samples=None,
    supervised_criterion=None,
    unsupervised_criterion=None,
    update_supervision=None,
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
        update_supervision=update_supervision,
    )
