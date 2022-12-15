import numpy as np
from sklearn.metrics import make_scorer, roc_curve, auc

def tpr_ppr_curve(
    y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True
):
    tpr, _, thresholds = roc_curve(
        y_true, y_score,
        pos_label=pos_label,
        sample_weight=sample_weight,
        drop_intermediate=drop_intermediate,
    )
    ppr = (y_score > thresholds[:, None]).mean(-1)

    return tpr, ppr, thresholds


def auc_tpr_ppr(
    y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True
):
    tpr, ppr, _ = roc_curve(
        y_true, y_score,
        pos_label=pos_label,
        sample_weight=sample_weight,
        drop_intermediate=drop_intermediate,
    )
    return auc(ppr, tpr)


auc_tpr_ppr_score = make_scorer(auc_tpr_ppr)