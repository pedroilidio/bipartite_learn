from typing import Type
import warnings
import copy
from numbers import Integral, Real
import numpy as np
from sklearn.tree._criterion import Criterion, ClassificationCriterion
from sklearn.tree._splitter import Splitter
from sklearn.utils.validation import check_random_state
from sklearn.utils._param_validation import (
    validate_params, Interval, HasMethods
)
from ._bipartite_splitter import BipartiteSplitter
from ._bipartite_criterion import GMOSA
from ._semisupervised_criterion import (
    SSCompositeCriterion,
    BipartiteSemisupervisedCriterion,
)
from ._axis_criterion import AxisCriterion
from . import _axis_criterion
from . import _unsupervised_criterion


AXIS_CRITERIA_CLF = {
    "gini": _axis_criterion.AxisGini,
    "entropy": _axis_criterion.AxisEntropy,
    "log_loss": _axis_criterion.AxisEntropy,
}
U_CATEGORIC_CRITERIA = {
    "gini": _unsupervised_criterion.UnsupervisedGini,
    "entropy": _unsupervised_criterion.UnsupervisedEntropy,
    "pairwise_gini": _unsupervised_criterion.PairwiseGini,
    "pairwise_entropy": _unsupervised_criterion.PairwiseEntropy,
}


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


def _is_classification_criterion_type(criterion: Type[Criterion]):
    return (
        criterion in AXIS_CRITERIA_CLF.values()
        or criterion in U_CATEGORIC_CRITERIA.values()
        or issubclass(criterion, ClassificationCriterion)
    )


def check_criterion(
    criterion: str | Criterion | Type[Criterion],
    *,
    n_samples: int | None = None,
    n_outputs: int | None = None,
    n_classes: int | None = None,
    kind: str | None = None,
    is_classification: bool | None = None,
    axis: bool = False,  # Useful only if isinstance(criterion, str)
) -> Criterion:

    if not isinstance(criterion, str) and kind is not None:
        warnings.warn(
            f"{kind=} parameter is being ignored since {criterion=} is "
            "not a string."
        )
    if isinstance(criterion, type):
        if not issubclass(criterion, (Criterion, AxisCriterion)):
            raise ValueError(
                f"Type {criterion=} provided is not a subclass of Criterion"
                " or AxisCriterion."
            )

        if is_classification is None:
            is_classification = _is_classification_criterion_type(criterion)

        if is_classification:
            if n_outputs is None or n_classes is None:
                raise ValueError(
                    "'n_outputs' and 'n_classes' must be provided if 'criterion' "
                    f"is a classification criterion type. Received {n_outputs=}, "
                    f"{n_classes=} and {criterion=}."
                )
            if len(n_classes) != n_outputs :
                raise ValueError(
                    "'len(n_classes)' must be 'n_outputs'. "
                    f"Received {n_outputs=} and {len(n_classes)=}. "
                    f"{n_classes=}."
                )
            final_criterion = criterion(
                n_outputs=n_outputs,
                n_classes=n_classes,
            )
        else:
            if n_outputs is None or n_samples is None:
                raise ValueError(
                    "'n_outputs' and 'n_samples' must be provided if 'criterion' "
                    f"is a regression criterion type. Received {n_outputs=}, "
                    f"{n_samples=} and {criterion=}."
                )
            final_criterion = criterion(
                n_outputs=n_outputs,
                n_samples=n_samples,
            )

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
#     bipartite_criterion_class=GMOSA,
# ))
def make_bipartite_splitter(
    splitters,
    criteria,
    *,
    is_classification=None,
    n_samples=None,
    n_classes=None,
    n_outputs=1,
    max_features=None,
    min_samples_leaf=1,
    min_weight_leaf=0.0,
    ax_min_samples_leaf=1,
    ax_min_weight_leaf=0.0,
    random_state=None,
    bipartite_criterion_class=GMOSA,
):
    """Factory function of BipartiteSplitter instances.

    Utility function to simplificate BipartiteSplitter instantiation.
    Parameters may be set to a single value or a 2-valued
    tuple or list, to specify them for each axis.

    ax_min_samples_leaf represents [min_rows_leaf, min_cols_leaf]
    """
    if criteria is not None:
        bipartite_criterion = make_bipartite_criterion(
            criteria,
            bipartite_criterion_class,
            is_classification=is_classification,
            n_samples=n_samples,
            n_classes=n_classes,
            n_outputs=n_outputs,
        )
        criteria = [
            bipartite_criterion.criterion_rows,
            bipartite_criterion.criterion_cols,
        ]

    (
        max_features,
        ax_min_samples_leaf,
        ax_min_weight_leaf,
        splitters,
    ) = _duplicate_single_parameters(
        max_features,
        ax_min_samples_leaf,
        ax_min_weight_leaf,
        splitters,
    )

    random_state = check_random_state(random_state)

    for ax in range(2):
        # Make splitter
        if issubclass(splitters[ax], Splitter):
            if criteria is None:
                raise ValueError (
                    f"Since splitters[{ax}] is a class, criteria must "
                    "be provided."
                )
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
        elif criteria is not None:
            warnings.warn(
                f"Since splitters[{ax}] is not a class, the provided "
                f"{criteria[ax]=} is being ignored."
            )

    # Wrap splitters.
    return BipartiteSplitter(
        splitter_rows=splitters[0],
        splitter_cols=splitters[1],
        bipartite_criterion=bipartite_criterion,
        min_samples_leaf=min_samples_leaf,
        min_weight_leaf=min_weight_leaf,
    )


def make_bipartite_criterion(
    criteria,
    bipartite_criterion_class,
    *,
    is_classification=None,
    n_samples=None,
    n_classes=None,
    n_outputs=1,
):
    criteria, n_samples, n_outputs = _duplicate_single_parameters(
        criteria,
        n_samples,
        n_outputs,
    )

    if n_classes is None:
        n_classes = (None, None)
    else:
        if len(n_classes) != 2:
            raise ValueError(
                "A list with n_classes for each axis must be provided in the "
                "n_classes parameter."
            )

    for ax in range(2):
        criteria[ax] = check_criterion(
            criteria[ax],
            n_samples=n_samples[ax],
            n_outputs=n_outputs[ax],
            n_classes=n_classes[ax],
            is_classification=is_classification,
        )

    return bipartite_criterion_class(criteria[0], criteria[1])


# TODO: docs
#     """Unsupervised impurity is only used to decide between rows or columns.
# 
#     The split search takes into consideration only the labels, as usual, but
#     after the rows splitter and the columns splitter defines each one's split,
#     unsupervised information is used to decide between them, i.e. the final
#     impurity is semisupervised as in GMOSA, but the proxy improvement
#     only uses supervised data.
#     """
def make_bipartite_ss_splitter(
    splitters: list[Type[Splitter]],
    supervised_criteria=None,  # FIXME: validate, cannot pass None
    unsupervised_criteria=None,
    ss_criteria=None,
    supervision=0.5,
    update_supervision=None,  # TODO: for each axis
    n_features=None,
    n_samples=None,
    n_classes=None,
    is_classification=None,
    max_features=None,
    n_outputs=1,
    min_samples_leaf=1,
    min_weight_leaf=0.0,
    ax_min_samples_leaf=1,
    ax_min_weight_leaf=0.0,
    random_state=None,
    axis_decision_only=False,
    bipartite_criterion_class=GMOSA,
    ss_bipartite_criterion_class=BipartiteSemisupervisedCriterion,
):
    """Factory function of BipartiteSplitter instances with semisupervised criteria.

    Utility function to simplificate BipartiteSplitter instantiation.
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

    bipartite_criterion = make_2dss_criterion(
        bipartite_criterion_class=bipartite_criterion_class,
        supervised_criteria=supervised_criteria,
        unsupervised_criteria=unsupervised_criteria,
        supervision=supervision,
        ss_criteria=ss_criteria,
        update_supervision=update_supervision,
        n_features=n_features,
        n_samples=n_samples,
        n_classes=n_classes,
        is_classification=is_classification,
        n_outputs=n_outputs,
        ss_bipartite_criterion_class=ss_bipartite_criterion_class,
    )

    if axis_decision_only:
        splitter_criteria = [
            bipartite_criterion.criterion_rows.supervised_criterion,
            bipartite_criterion.criterion_cols.supervised_criterion,
        ]
    else:
        splitter_criteria = [
            bipartite_criterion.criterion_rows,
            bipartite_criterion.criterion_cols,
        ]

    for ax in range(2):
        if not issubclass(splitters[ax], Splitter):
            raise TypeError

        splitters[ax] = splitters[ax](
            criterion=splitter_criteria[ax],
            max_features=max_features[ax],
            min_samples_leaf=ax_min_samples_leaf[ax],
            min_weight_leaf=ax_min_weight_leaf[ax],
            random_state=random_state,
        )

    return BipartiteSplitter(
        splitter_rows=splitters[0],
        splitter_cols=splitters[1],
        bipartite_criterion=bipartite_criterion,
        min_samples_leaf=min_samples_leaf,
        min_weight_leaf=min_weight_leaf,
    )


@validate_params(dict(
    supervision=[Interval(Real, 0.0, 1.0, closed="both")],
    supervised_criterion=[Criterion, type(Criterion)],
    unsupervised_criterion=[Criterion, type(Criterion)],
    is_classification=["boolean", None],
    n_outputs=[Interval(Integral, 1, None, closed="left"), None],
    n_features=[Interval(Integral, 1, None, closed="left"), None],
    n_samples=[Interval(Integral, 1, None, closed="left"), None],
    n_classes=[np.ndarray, None],
    update_supervision=[callable, None],
    ss_class=[type(SSCompositeCriterion), None],
))
def make_semisupervised_criterion(
    *,
    supervision,
    supervised_criterion,
    unsupervised_criterion,
    is_classification=None,
    n_classes=None,
    n_outputs=None,
    n_features=None,
    n_samples=None,
    update_supervision=None,
    ss_class=None,
):
    supervised_criterion = check_criterion(
        criterion=supervised_criterion,
        n_outputs=n_outputs,
        n_samples=n_samples,
        n_classes=n_classes,
        is_classification=is_classification,
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
    )
    if isinstance(unsupervised_criterion, AxisCriterion):
        raise TypeError(
            f"{unsupervised_criterion=} cannot be an instance of AxisCriterion"
        )

    if ss_class is None:
        ss_class = SSCompositeCriterion

    return ss_class(
        supervision=supervision,
        supervised_criterion=supervised_criterion,
        unsupervised_criterion=unsupervised_criterion,
        update_supervision=update_supervision,
    )


def make_2dss_criterion(
    bipartite_criterion_class,
    supervised_criteria,
    unsupervised_criteria,
    supervision,
    ss_criteria=None,
    update_supervision=None,  # TODO: for each axis
    n_features=None,
    n_samples=None,
    n_classes=None,
    n_outputs=1,
    is_classification=None,
    ss_bipartite_criterion_class=BipartiteSemisupervisedCriterion,
):
    """Factory function of BipartiteSplitter instances with semisupervised criteria.

    Utility function to simplificate BipartiteSplitter instantiation.
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
    ) = _duplicate_single_parameters(
        n_samples,
        n_features,
        n_outputs,
        supervision,
        ss_criteria,
        supervised_criteria,
        unsupervised_criteria,
    )

    if n_classes is None:
        n_classes = (None, None)

    # Make semi-supervised criteria
    for ax in range(2):
        if not isinstance(ss_criteria[ax], SSCompositeCriterion):
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
                n_classes=n_classes[ax],
                is_classification=is_classification,
            )
        elif (
            supervised_criteria[ax] is not None
            or unsupervised_criteria[ax] is not None
        ):
            raise ValueError(
                "ss_criteria is not a type but supervised or unsupervised "
                f"criteria provided:\n  {supervised_criteria[ax]=}"
                f"\n  {unsupervised_criteria[ax]=}"
                f"\n  {ss_criteria[ax]=}"
            )
    
    return ss_bipartite_criterion_class(
        bipartite_criterion=bipartite_criterion_class(
            criterion_rows=ss_criteria[0],
            criterion_cols=ss_criteria[1],
        )
    )
