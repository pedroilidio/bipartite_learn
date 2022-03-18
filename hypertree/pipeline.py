"""
Adapts `sklearn.pipeline.Pipeline` to transformers that yield new y instead of
new X (TargetPipeline) and to transformers that return both new X and new y
(XYPipeline). In both cases, the transformers may receive X and y in their fit
methods, which is not possible with `sklearn.compose.TransformedTargetRegressor`.
Changes are marked with `###`.
"""
# Author: Pedro Ilídio
# License: BSD


from sklearn.pipeline import (
    Pipeline, check_memory,
    _fit_transform_one,
    _print_elapsed_time,
    _final_estimator_has,
)
from sklearn.utils.metaestimators import available_if


class TargetPipeline(Pipeline):
    """Works the same way a normal Pipeline, but passes along y instead of X."""
    def _fit(self, X, y=None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for (step_idx, name, transformer) in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("TargetPipeline", self._log_message(step_idx)):  ###
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transformer
            y, fitted_transformer = fit_transform_one_cached(
            ### X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="TargetPipeline",  ###
                message=self._log_message(step_idx),
                **fit_params_steps[name],
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return y
        ### return X

    def fit(self, X, y=None, **fit_params):
        """Fit the model.
        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements for all steps of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements of first step of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        self : object
            Pipeline with fitted steps.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        yt = self._fit(X, y, **fit_params_steps)
        ### Xt = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(X, yt, **fit_params_last_step)
                ### self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator.
        Fits all the transformers one after the other and transform the
        data. Then uses `fit_transform` on transformed data with the final
        estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements for all steps of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements of first step of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        yt = self._fit(X, y, **fit_params_steps)
        ### Xt = self._fit(X, y, **fit_params_steps)

        last_step = self._final_estimator
        with _print_elapsed_time("TargetPipeline", self._log_message(len(self.steps) - 1)):  ###
            if last_step == "passthrough":
                return yt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(X, yt, **fit_params_last_step)
                ### return last_step.fit_transform(Xt, y, **fit_params_last_step)
            else:
                return last_step.fit(X, yt, **fit_params_last_step).transform(X, yt)
                ### return last_step.fit(Xt, y, **fit_params_last_step).transform(Xt)

    @available_if(_final_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        """Transform the data, and apply `fit_predict` with the final estimator.
        Call `fit_transform` of each transformer in the pipeline. The
        transformed data are finally passed to the final estimator that calls
        `fit_predict` method. Only valid if the final estimator implements
        `fit_predict`.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        y_pred : ndarray
            Result of calling `fit_predict` on the final estimator.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        yt = self._fit(X, y, **fit_params_steps)
        ### Xt = self._fit(X, y, **fit_params_steps)

        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            y_pred = self.steps[-1][1].fit_predict(X, yt, **fit_params_last_step)
            ### y_pred = self.steps[-1][1].fit_predict(Xt, y, **fit_params_last_step)
        return y_pred


class XYPipeline(Pipeline):
    """Works the same way a normal Pipeline, but passes along y instead of X."""
    def _fit(self, X, y=None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for (step_idx, name, transformer) in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("XYPipeline", self._log_message(step_idx)):  ###
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transformer
            (X, y), fitted_transformer = fit_transform_one_cached(
            ### X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="XYPipeline",  ###
                message=self._log_message(step_idx),
                **fit_params_steps[name],
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X, y
        ### return X

    def fit(self, X, y=None, **fit_params):
        """Fit the model.
        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements of first step of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        self : object
            Pipeline with fitted steps.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)
        ### Xt = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time("XYPipeline", self._log_message(len(self.steps) - 1)):  ###
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, yt, **fit_params_last_step)
                ### self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator.
        Fits all the transformers one after the other and transform the
        data. Then uses `fit_transform` on transformed data with the final
        estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements of first step of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)
        ### Xt = self._fit(X, y, **fit_params_steps)

        last_step = self._final_estimator
        with _print_elapsed_time("XYPipeline", self._log_message(len(self.steps) - 1)):  ###
            if last_step == "passthrough":
                return Xt, yt
                ### return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(Xt, yt, **fit_params_last_step)
                ### return last_step.fit_transform(Xt, y, **fit_params_last_step)
            else:
                return last_step.fit(Xt, yt, **fit_params_last_step).transform(Xt, yt)
                ### return last_step.fit(Xt, y, **fit_params_last_step).transform(Xt)

    @available_if(_final_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        """Transform the data, and apply `fit_predict` with the final estimator.
        Call `fit_transform` of each transformer in the pipeline. The
        transformed data are finally passed to the final estimator that calls
        `fit_predict` method. Only valid if the final estimator implements
        `fit_predict`.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        y_pred : ndarray
            Result of calling `fit_predict` on the final estimator.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)
        ### Xt = self._fit(X, y, **fit_params_steps)

        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            y_pred = self.steps[-1][1].fit_predict(Xt, yt, **fit_params_last_step)
            ### y_pred = self.steps[-1][1].fit_predict(Xt, y, **fit_params_last_step)
        return y_pred
