from typing import Type
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
from ._pairwise_criterion import PairwiseCriterion


def _duplicate_single_parameters(*params, ndim=2):
    result = []

    for param in params:
        if isinstance(param, (list, tuple)):
            if len(param) != ndim:
                raise ValueError(
                    f"param {param=} has length {len(param)}, expected {ndim}."
                )
            result.append(list(param))
        else:
            result.append([copy.deepcopy(param) for _ in range(ndim)])

    return result


def check_criterion(
    criterion: str | Criterion | Type[Criterion],
    n_samples: int | None = None,
    n_outputs: int | None = None,
    kind: str | None = None,
    classification: bool = False,
    pairwise: bool = False,
    axis: bool = False,  # Useful with str criterion only
) -> Criterion:

    if not isinstance(criterion, str) and kind is not None:
        warnings.warn(
            f"{kind=} parameter is being ignored since {criterion=} is "
            "not a string."
        )
    if isinstance(criterion, type):
        if n_samples is None or n_outputs is None:
            raise ValueError(
                "'n_samples' and 'n_outputs' must be provided if 'criterion' "
                f"is a type. Received {n_samples=}, {n_outputs=} and "
                f"{criterion=}."
            )
        if not issubclass(criterion, Criterion):
            raise ValueError(
                f"Type {criterion=} provided is not a subclass of Criterion."
            )
        final_criterion = criterion(
            n_outputs=n_outputs,
            n_samples=n_samples,
        )
        if pairwise:
            final_criterion = PairwiseCriterion(final_criterion)

        return final_criterion

    elif isinstance(criterion, str):
        raise NotImplementedError
    
    elif isinstance(criterion, Criterion):
        if n_samples is not None:
            warnings.warn(
                f"{n_samples=} parameter is being ignored since {criterion=} "
                "is already a Criterion instance."
            )
        if n_outputs is not None:
            warnings.warn(
                f"{n_outputs=} parameter is being ignored since {criterion=} "
                "is already a Criterion instance."
            )
        return criterion
    else:  # Should never run because of @validate_params
        raise ValueError(f"Unknown criterion type ({criterion=}).")


# TODO:
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
        criteria[ax] = check_criterion(
            criteria[ax],
            n_samples=n_samples[ax],
            n_outputs=n_outputs[ax],
        )

        # Make splitter
        if issubclass(splitters[ax], Splitter):
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
    criterion_wrapper = criterion_wrapper_class(criteria[0], criteria[1])

    # Wrap splitters.
    return Splitter2D(
        splitter_rows=splitters[0],
        splitter_cols=splitters[1],
        criterion_wrapper=criterion_wrapper,
        min_samples_leaf=min_samples_leaf,
        min_weight_leaf=min_weight_leaf,
    )


# TODO: docs
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
    update_supervision=None,  # TODO: for each axis
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
        max_features,
        ax_min_samples_leaf,
        ax_min_weight_leaf,
        splitters,
        supervised_criteria,
        unsupervised_criteria,
        ss_criteria,
    ) = _duplicate_single_parameters(
        max_features,
        ax_min_samples_leaf,
        ax_min_weight_leaf,
        splitters,
        supervised_criteria,
        unsupervised_criteria,
        ss_criteria,
    )

    random_state = check_random_state(random_state)

    # Make semi-supervised criteria
    for ax in range(2):
        if isinstance(splitters[ax], Splitter):
            if (supervised_criteria is not None):
                warnings.warn(
                    f"Since {splitters[ax]=} is not a class, the provided "
                    f"{supervised_criteria=} is being ignored."
                )
            if (unsupervised_criteria is not None):
                warnings.warn(
                    f"Since {splitters[ax]=} is not a class, the provided "
                    f"{unsupervised_criteria=} is being ignored."
                )
            if axis_decision_only:
                supervised_criteria[ax] = splitters[ax].criterion
            else:
                # FIXME: no access to Cython private attributes
                ss_criteria[ax] = splitters[ax].criterion
                supervised_criteria[ax] = ss_criteria[ax].supervised_criterion
                unsupervised_criteria[ax] = ss_criteria[ax].unsupervised_criterion

    criterion_wrapper = make_2dss_criterion(
        criterion_wrapper_class=criterion_wrapper_class,
        supervised_criteria=supervised_criteria,
        unsupervised_criteria=unsupervised_criteria,
        supervision=supervision,
        ss_criteria=ss_criteria,
        update_supervision=update_supervision,
        n_features=n_features,
        n_samples=n_samples,
        n_outputs=n_outputs,
        random_state=random_state,
        pairwise=pairwise,
        axis_decision_only=axis_decision_only,
        ss_criterion_wrapper_class=ss_criterion_wrapper_class,
    )

    for ax in range(2):
        if issubclass(splitters[ax], Splitter):
            if axis_decision_only:
                splitter_criteria = [
                    criterion_wrapper.supervised_criterion_rows,
                    criterion_wrapper.supervised_criterion_cols,
                ]
            else:
                splitter_criteria = [
                    criterion_wrapper.ss_criterion_rows,
                    criterion_wrapper.ss_criterion_cols,
                ]

            splitters[ax] = splitters[ax](
                criterion=splitter_criteria[ax],
                max_features=max_features[ax],
                min_samples_leaf=ax_min_samples_leaf[ax],
                min_weight_leaf=ax_min_weight_leaf[ax],
                random_state=random_state,
            )

    return Splitter2D(
        splitter_rows=splitters[0],
        splitter_cols=splitters[1],
        criterion_wrapper=criterion_wrapper,
        min_samples_leaf=min_samples_leaf,
        min_weight_leaf=min_weight_leaf,
    )


@validate_params(dict(
    supervised_criterion=[Criterion, type(Criterion)],
    unsupervised_criterion=[Criterion, type(Criterion)],
    supervision=[Interval(Real, 0.0, 1.0, closed="both")],
    n_outputs=[Interval(Integral, 1, None, closed="left"), None],
    n_features=[Interval(Integral, 1, None, closed="left"), None],
    n_samples=[Interval(Integral, 1, None, closed="left"), None],
    update_supervision=[callable, None],
    ss_class=[type(SemisupervisedCriterion), None],
    pairwise=["boolean"],
))
def make_semisupervised_criterion(
    supervision,
    supervised_criterion,
    unsupervised_criterion,
    n_outputs=None,
    n_features=None,
    n_samples=None,
    update_supervision=None,
    ss_class=SSCompositeCriterion,
    pairwise=False,
):
    # Facilitate setting the default ss_class as simply None
    ss_class = SSCompositeCriterion if ss_class is None else ss_class

    supervised_criterion = check_criterion(
        criterion=supervised_criterion,
        n_outputs=n_outputs,
        n_samples=n_samples,
    )

    # Catch before check_criterion() to make it clear that it was the
    # n_features which was missing.
    if isinstance(unsupervised_criterion, type):
        if not n_features or not n_samples:
            raise ValueError(
                'If unsupervised_criterion is a class, one must provide both '
                f'n_features (received {n_features}) and n_samples ({n_samples}).'
            )

    unsupervised_criterion = check_criterion(
        criterion=unsupervised_criterion,
        n_outputs=n_features,
        n_samples=n_samples,
        pairwise=pairwise,
    )

    return ss_class(
        supervision=supervision,
        supervised_criterion=supervised_criterion,
        unsupervised_criterion=unsupervised_criterion,
        update_supervision=update_supervision,
    )


def make_2dss_criterion(
    criterion_wrapper_class,
    supervised_criteria,
    unsupervised_criteria,
    supervision,
    ss_criteria=None,
    update_supervision=None,  # TODO: for each axis
    n_features=None,
    n_samples=None,
    n_outputs=1,
    random_state=None,
    pairwise=False,
    axis_decision_only=False,
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
        n_outputs,
        supervision,
        ss_criteria,
        supervised_criteria,
        unsupervised_criteria,
        pairwise,
    ) = _duplicate_single_parameters(
        n_samples,
        n_features,
        n_outputs,
        supervision,
        ss_criteria,
        supervised_criteria,
        unsupervised_criteria,
        pairwise,
    )

    random_state = check_random_state(random_state)

    # Make semi-supervised criteria
    for ax in range(2):
        if axis_decision_only:  
            supervised_criteria[ax] = check_criterion(
                supervised_criteria[ax],
                n_outputs=n_outputs[ax],
                n_samples=n_samples[ax],
            )
            unsupervised_criteria[ax] = check_criterion(
                unsupervised_criteria[ax],
                n_outputs=n_features[ax],
                n_samples=n_samples[ax],
                pairwise=pairwise[ax],
            )

            if (ss_criteria[ax] is not None):
                warnings.warn(
                    f"Since {axis_decision_only=} is not None, the provided "
                    f"{ss_criteria[ax]=} is being ignored."
                )
        else:  # not axis_decision_only
            if ss_criteria[ax] is None:
                ss_criteria[ax] = SSCompositeCriterion

            if not isinstance(ss_criteria[ax], SemisupervisedCriterion):
                ss_criteria[ax] = make_semisupervised_criterion(
                    ss_class=ss_criteria[ax],
                    supervision=supervision[ax],
                    # update_supervision=update_supervision,
                    update_supervision=None,  # The wrapper will take charge
                    supervised_criterion=supervised_criteria[ax],
                    unsupervised_criterion=unsupervised_criteria[ax],
                    n_outputs=n_outputs[ax],
                    n_features=n_features[ax],
                    n_samples=n_samples[ax],
                    pairwise=pairwise[ax],
                )
            
            supervised_criteria[ax] = ss_criteria[ax].supervised_criterion
            unsupervised_criteria[ax] = ss_criteria[ax].unsupervised_criterion
    
    return ss_criterion_wrapper_class(
        unsupervised_criterion_rows=unsupervised_criteria[0],
        unsupervised_criterion_cols=unsupervised_criteria[1],
        supervised_bipartite_criterion=criterion_wrapper_class(
            criterion_rows=supervised_criteria[0],
            criterion_cols=supervised_criteria[1],
        ),
        supervision_rows=supervision[0],
        supervision_cols=supervision[1],
        update_supervision=update_supervision,
        ss_criterion_rows=ss_criteria[0],
        ss_criterion_cols=ss_criteria[1],
    )


# def make_2dss_splitter(
#     splitters,
#     supervised_criteria=None,  # FIXME: validate, cannot pass None
#     unsupervised_criteria=None,
#     ss_criteria=None,
#     supervision=0.5,
#     update_supervision=None,  # TODO: for each axis
#     n_features=None,
#     n_samples=None,
#     max_features=None,
#     n_outputs=1,
#     min_samples_leaf=1,
#     min_weight_leaf=0.0,
#     ax_min_samples_leaf=1,
#     ax_min_weight_leaf=0.0,
#     random_state=None,
#     pairwise=False,
#     axis_decision_only=False,
#     criterion_wrapper_class=MSE_Wrapper2D,
#     ss_criterion_wrapper_class=BipartiteSemisupervisedCriterion,
# ):
#     """Factory function of Splitter2D instances with semisupervised criteria.
# 
#     Utility function to simplificate Splitter2D instantiation.
#     Parameters may be set to a single value or a 2-valued
#     tuple or list, to specify them for each axis.
#     """
#     (
#         n_samples,
#         n_features,
#         max_features,
#         n_outputs,
#         supervision,
#         ax_min_samples_leaf,
#         ax_min_weight_leaf,
#         splitters,
#         ss_criteria,
#         supervised_criteria,
#         unsupervised_criteria,
#         pairwise,
#     ) = _duplicate_single_parameters(
#         n_samples,
#         n_features,
#         max_features,
#         n_outputs,
#         supervision,
#         ax_min_samples_leaf,
#         ax_min_weight_leaf,
#         splitters,
#         ss_criteria,
#         supervised_criteria,
#         unsupervised_criteria,
#         pairwise,
#     )
# 
#     random_state = check_random_state(random_state)
# 
#     # Make semi-supervised criteria
#     for ax in range(2):
#         if isinstance(splitters[ax], Splitter):
#             if (supervised_criteria[ax] is not None):
#                 warnings.warn(
#                     f"Since {splitters[ax]=} is not a class, the provided "
#                     f"{supervised_criteria[ax]=} is being ignored."
#                 )
#             if (unsupervised_criteria[ax] is not None):
#                 warnings.warn(
#                     f"Since {splitters[ax]=} is not a class, the provided "
#                     f"{unsupervised_criteria[ax]=} is being ignored."
#                 )
#             continue
#         # TODO: validate params
#         elif not issubclass(splitters[ax], Splitter):  # TODO: use duck typing?
#             raise ValueError(
#                 "splitter must be an instance or subclass of "
#                 "sklearn.tree._splitter.Splitter"
#             )
#         if axis_decision_only:  
#             supervised_criteria[ax] = check_criterion(
#                 supervised_criteria[ax],
#                 n_outputs=n_outputs[ax],
#                 n_samples=n_samples[ax],
#             )
#             unsupervised_criteria[ax] = check_criterion(
#                 unsupervised_criteria[ax],
#                 n_outputs=n_features[ax],
#                 n_samples=n_samples[ax],
#                 pairwise=pairwise[ax],
#             )
# 
#             if (ss_criteria[ax] is not None):
#                 warnings.warn(
#                     f"Since {axis_decision_only=} is not None, the provided "
#                     f"{ss_criteria[ax]=} is being ignored."
#                 )
#             # The Splitter will not consider unsupevised data when searching
#             # for a split.
#             splitters[ax] = splitters[ax](
#                 criterion=supervised_criteria[ax],
#                 max_features=max_features[ax],
#                 min_samples_leaf=ax_min_samples_leaf[ax],
#                 min_weight_leaf=ax_min_weight_leaf[ax],
#                 random_state=random_state,
#             )
# 
#         else:
#             if ss_criteria[ax] is None:
#                 ss_criteria[ax] = SSCompositeCriterion
# 
#             if not isinstance(ss_criteria[ax], SemisupervisedCriterion):
#                 ss_criteria[ax] = make_semisupervised_criterion(
#                     ss_class=ss_criteria[ax],
#                     supervision=supervision[ax],
#                     # update_supervision=update_supervision,
#                     update_supervision=None,  # The wrapper will take charge
#                     supervised_criterion=supervised_criteria[ax],
#                     unsupervised_criterion=unsupervised_criteria[ax],
#                     n_outputs=n_outputs[ax],
#                     n_features=n_features[ax],
#                     n_samples=n_samples[ax],
#                     pairwise=pairwise[ax],
#                 )
#             
#             supervised_criteria[ax] = ss_criteria[ax].supervised_criterion
#             unsupervised_criteria[ax] = ss_criteria[ax].unsupervised_criterion
# 
#             splitters[ax] = splitters[ax](
#                 criterion=ss_criteria[ax],
#                 max_features=max_features[ax],
#                 min_samples_leaf=ax_min_samples_leaf[ax],
#                 min_weight_leaf=ax_min_weight_leaf[ax],
#                 random_state=random_state,
#             )
#     
#     criterion_wrapper = ss_criterion_wrapper_class(
#         unsupervised_criterion_rows=unsupervised_criteria[0],
#         unsupervised_criterion_cols=unsupervised_criteria[1],
#         supervised_bipartite_criterion=criterion_wrapper_class(
#             criterion_rows=supervised_criteria[0],
#             criterion_cols=supervised_criteria[1],
#         ),
#         supervision_rows=supervision[0],
#         supervision_cols=supervision[1],
#         update_supervision=update_supervision,
#         ss_criterion_rows=ss_criteria[0],
#         ss_criterion_cols=ss_criteria[1],
#     )
# 
#     return Splitter2D(
#         splitter_rows=splitters[0],
#         splitter_cols=splitters[1],
#         criterion_wrapper=criterion_wrapper,
#         min_samples_leaf=min_samples_leaf,
#         min_weight_leaf=min_weight_leaf,
#     )
