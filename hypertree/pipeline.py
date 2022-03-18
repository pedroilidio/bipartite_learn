"""
Adapts `sklearn.pipeline.Pipeline` to transformers that yield new y instead of
new X (TargetPipeline) and to transformers that return both new X and new y
(XYPipeline). In both cases, the transformers may receive X and y in their fit
methods, which is not possible with `sklearn.compose.TransformedTargetRegressor`.
"""
# Author: Pedro Ilídio
# License: BSD


from sklearn.pipeline import \
    Pipeline, check_memory, _fit_transform_one, _print_elapsed_time


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
        fit_params_steps = self._check_fit_params(**fit_params)
        yt = self._fit(X, y, **fit_params_steps)

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
