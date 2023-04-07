"""Distance Weighted Neighbors Regression."""

# Author: Pedro Ilídio <ilidio@alumni.usp.br>
# Adapted from scikit-learn.

# License: BSD 3 clause (C) Pedro Ilídio

import numpy as np

from joblib import effective_n_jobs
from sklearn.neighbors._base import (
    _get_weights, _check_precomputed, NeighborsBase, KNeighborsMixin,
)
from sklearn.base import RegressorMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils._param_validation import StrOptions


class WeightedNeighborsRegressor(KNeighborsMixin, RegressorMixin, NeighborsBase):
    """Regression based on distance-weighted neighbors.
    The target is predicted by distance-weighted interpolation of the targets
    associated with the instances in the training set. Corresponds to
    `KneighborsRegressor(n_neighbors=X_train.shape[0])`, but much faster.
    Parameters
    ----------
    weights : 'distance', callable or None, default='distance'
        Weight function used in prediction.  Possible values:
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.
        Distance weights are used by default.
    p : int, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors.
        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.
    Attributes
    ----------
    effective_metric_ : str or callable
        The distance metric to use. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.
    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    n_samples_fit_ : int
        Number of samples in the fitted data.
    See Also
    --------
    NearestNeighbors : Unsupervised learner for implementing neighbor searches.
    RadiusNeighborsRegressor : Regression based on neighbors within a fixed radius.
    KNeighborsRegressor : Classifier implementing the k-nearest neighbors vote.
    KNeighborsClassifier : Classifier implementing the k-nearest neighbors vote.
    RadiusNeighborsClassifier : Classifier implementing
        a vote among neighbors within a given radius.
    Examples
    --------
    >>> X = [[0], [1], [2], [3]]
    >>> y = [0, 0, 1, 1]
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> neigh = KNeighborsRegressor(n_neighbors=2)
    >>> neigh.fit(X, y)
    KNeighborsRegressor(...)
    >>> print(neigh.predict([[1.5]]))
    [0.5]
    """

    _parameter_constraints: dict = {
        **NeighborsBase._parameter_constraints,
        "weights": [StrOptions({"distance"}), callable, None],
    }
    _parameter_constraints.pop("radius")
    _parameter_constraints.pop("n_neighbors")
    _parameter_constraints.pop("leaf_size")
    _parameter_constraints.pop("algorithm")

    def __init__(
        self,
        *,
        weights="distance",
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=None,
            algorithm="brute",
            leaf_size=None,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.weights = weights

    def _more_tags(self):
        # For cross-validation routines to split data correctly
        return {"pairwise": self.metric == "precomputed"}

    def fit(self, X, y):
        """Fit the k-nearest neighbors regressor from the training dataset.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.
        y : {array-like, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_outputs)
            Target values.
        Returns
        -------
        self : KNeighborsRegressor
            The fitted k-nearest neighbors regressor.
        """
        self._validate_params()
        self.n_neighbors = X.shape[0]  # TODO: optimize other methods that use it
        return self._fit(X, y)

    def predict(self, X):
        """Predict the target for the provided data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), dtype=int
            Target values.
        """
        if self.metric == "precomputed":
            dist = _check_precomputed(X)
        else:
            X = self._validate_data(
                X, accept_sparse="csr", reset=False, order="C",
            )

            dist = pairwise_distances(
                X,
                self._fit_X,
                metric=self.effective_metric_,
                n_jobs=effective_n_jobs(self.n_jobs),
                **self.effective_metric_params_,
            )

        # Default weights is "distance", no more "uniform"
        weights = _get_weights(dist, self.weights or "distance")
        weights /= np.sum(weights, axis=1, keepdims=True)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        y_pred = weights @ _y

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred
