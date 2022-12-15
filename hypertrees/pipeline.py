from sklearn.utils import _safe_tags
from imblearn.pipeline import make_pipeline
from .wrappers import MultipartiteTransformerWrapper


# TODO: test get/set_params
def make_multipartite_pipeline(*steps, ndims=2, memory=None, verbose=False):
    """Utility function to create pipelines for multipartite data.

    It wraps monopartite transformers with MultipartiteTransformerWrapper.
    """
    # Make pipeline before modifying to preserve step names
    pipe = make_pipeline(*steps, memory=memory, verbose=verbose)

    for i, step in enumerate(pipe.steps[:-1]):
        if not _safe_tags(step, "multipartite"):
            pipe.steps[i] = MultipartiteTransformerWrapper(step, ndims=ndims)
