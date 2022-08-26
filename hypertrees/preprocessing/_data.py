from copy import deepcopy
from sklearn.base import TransformerMixin


class TransformerWrapperND(TransformerMixin):
    def __init__(self, transformers, ndim=2) -> None:
        if isinstance(transformers, (list, tuple)):
            self.ndim = len(transformers)
            self.transformers = transformers
        else:
            self.transformers = [deepcopy(transformers) for i in range(ndim)]
            self.ndim = ndim

    def fit(self, X, y=None):
        self._validate_input(X, y)
        for Xi, transformer in zip(X, self.transformers):
            transformer.fit(X=Xi)

    def transform(self, X, y=None):
        self._validate_input(X, y)
        return [
            transformer.transform(X=Xi)
            for Xi, transformer in zip(X, self.transformers)
        ]

    def fit_transform(self, X, y=None):
        self._validate_input(X, y)
        return [
            transformer.fit_transform(X=Xi)
            for Xi, transformer in zip(X, self.transformers)
        ]

    def _validate_input(self, X, y):
        if len(X) != self.ndim:
            raise ValueError(f"Wrong dimension. X has {len(X)} items, was exp"
                             "ecting {self.ndim}.")
