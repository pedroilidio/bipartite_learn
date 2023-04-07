from sklearn.utils._tags import _safe_tags
from imblearn.pipeline import make_pipeline
from .wrappers import (
    MultipartiteTransformerWrapper,
    MultipartiteSamplerWrapper,
)


# TODO: test get/set_params
def make_multipartite_pipeline(*steps, ndim=2, memory=None, verbose=False):
    """Utility function to create pipelines for multipartite data.

    It wraps monopartite transformers with MultipartiteTransformerWrapper.
    """
    # Make pipeline before modifying to preserve step names
    pipe = make_pipeline(*steps, memory=memory, verbose=verbose)

    # FIXME: what if {"multipartite": False}?
    if "multipartite" not in _safe_tags(pipe.steps[-1][1]):
        raise ValueError(
            "Last step of a multipartite pipeline must be a multipartite "
            f"estimator, not {pipe.steps[-1][1]}."
        )

    for i, step in enumerate(pipe.steps[:-1]):
        # FIXME: what if {"multipartite": False}?
        if "multipartite" not in _safe_tags(step[1]):
            estimator = step[1]
            if (
                hasattr(estimator, "_estimator_type")
                and estimator._estimator_type == "sampler"
            ):
                wrapped = MultipartiteSamplerWrapper(estimator, ndim=ndim)
            elif hasattr(estimator, "transform"):
                wrapped = MultipartiteTransformerWrapper(estimator, ndim=ndim)
            else:
                raise ValueError(
                    f"Pipeline step of type {type(estimator)} is not a sampler"
                    " or transformer."
                )

            pipe.steps[i] = (step[0], wrapped)
    
    return pipe
