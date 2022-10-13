# TODO: move to wrappers?
from __future__ import annotations
from typing import Sequence
from sklearn.base import BaseEstimator, TransformerMixin, clone
from hypertrees.base import BaseMultipartiteEstimator


# FIXME: test get_params()
class MultipartiteTransformerWrapper(BaseMultipartiteEstimator,
                                     TransformerMixin):
    """Manages a transformer for each feature matrix (X) in bipartite sets.
    """
    def __init__(
        self,
        transformers : BaseEstimator | Sequence[BaseEstimator],
        ndim : int | None = 2,
    ):
        self.transformers = transformers
        self.ndim = ndim

    def fit(self, X, y=None):
        self._set_transformers()
        self._validate_input(X, y)

        for Xi, transformer in zip(X, self.transformers_):
            transformer.fit(X=Xi)

    def transform(self, X, y=None):
        self._validate_input(X, y)
        return [
            transformer.transform(X=Xi)
            for Xi, transformer in zip(X, self.transformers_)
        ]

    def fit_transform(self, X, y=None):
        self._set_transformers()
        self._validate_input(X, y)
        return [
            transformer.fit_transform(X=Xi)
            for Xi, transformer in zip(X, self.transformers_)
        ]

    def _set_transformers(self):
        """Sets self.transformers_ and self.ndim_ during fit.
        """
        if isinstance(self.transformers, Sequence):
            if self.ndim is None:
                self.ndim_ = len(self.transformers)

            elif self.ndim != len(self.transformers):
                raise ValueError("'self.ndim' must correspond to the number of"
                                 " transformers in self.transformers")
            else:
                self.ndim_ = self.ndim

            self.transformers_ = [clone(t) for t in self.transformers]

        else:  # If a single transformer provided
            if self.ndim is None:
                raise ValueError("'ndim' must be provided if 'transformers' is"
                                 " a single transformer.")
            self.ndim_ = self.ndim
            self.transformers_ = [
                clone(self.transformers) for _ in range(self.ndim_)
            ]

    def _validate_input(self, X, y):
        if len(X) != self.ndim_:
            raise ValueError(f"Wrong dimension. X has {len(X)} items, was exp"
                             "ecting {self.ndim}.")
