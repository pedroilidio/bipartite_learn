# bipartite_learn tutorial

_Machine learning estimators tailored to bipartite datasets._

In a usual machine learning setting, one is interested in predicting a set of
outputs `y` from a given feature vector `x` representing an input instance.
There are tasks, however, that are sometimes better modeled as bipartite
networks, in which two domains of instances are present and only inter-domain
relationships are plausible between pairs of instances. The goal is then to
predict aspects (`y`) of such interaction between a sample from the first domain
and another from the second domain, respectively represented by feature vectors
`x1` and `x2`.  In other words, it is sometimes desirable to model a function
in the format `(x1, x2) -> y` rather than the usual `x -> y` format.

Examples of such tasks can be found in the realms of interaction prediction and recommendation systems, and the datasets corresponding to them can be presented
as a pair of design matrices (`X1` and `X2`) together with an interaction
matrix `Y` that describes each relationship between the samples `X1[i]` and 
`X2[j]` in the position `Y[i, j]`.

This package provides:

1. A collection of tools to adapt usual algorithms to bipartite data;
2. Tree-based estimators designed specifically to such
datasets, which yield expressive performance improvements over the naive
adaptations of their monopartite counterparts.

## Installation
`bipartite_learn` is available on PyPI, and thus can be installed with `pip`:
```
$ pip install bipartite_learn
```
Local installation can be done either by providing the `--user` flag to the above command or by cloning this repository and issuing `pip` afterwards.
```
$ git clone https://github.com/pedroilidio/bipartite_learn
$ cd bipartite_learn
$ pip install --editable .
```
The optional `--editable` (or `-e`) flag installs the package as symbolic links
to the local cloned repository, so that changes in it will be immediatly
recognized.

{install github sklearn}

This package builds on the great scikit-learn API.

In their `fit` methods, instead of receiving a single `X` feature matrix and a
`y` target matrix, the
bipartite estimators in this package are built to receive two matrices `X1`
and `X2` wrapped in a list `[X1, X2]` as `fit`'s `X` parameter, along with an
interaction matrix `y` such that
`y.shape == (X[0].shape[0], X[1].shape[0])` and `y[i, j]` is the target
information to be predicted about the interaction between the samples `X[0][i]`
and `X[1][j]`.

Notice that although `y` is 2D, which would represent a
multioutput task for scikit-learn estimators, the target matrix `y` in bipartite
datasets essentially represents a single output per interacting pair. These
interactions themselves (the edges of the bipartite graph) are what we actually
consider the input samples, so that bipartite estimators are still considered
single-output estimators even if dealing with a bidimensional `y`. They can be
viewed as simpler (and often more efficient) methods for considering all
possible relationships during the training procedure, without the need for
explicit data preprocessing.

> Note: multioutput bipartite tasks are not supported by `bipartite_learn` for
now. In such cases, `y` would be most naturally represented as a
tridimensional tensor, storing each output value along its last dimension
(the "depth").

Even if essentially single-output under the eyes of `bipartite_learn`, some
learning algorithms for bipartite data make use of compositions of
multioutput traditional estimators (that are designed to be trained on a single
`X` matrix).

> Note: for clarity sake, we refer to the usual machine learning estimators
taking a single `X` matrix as _monopartite_ estimators, while the ones that we
mainly focus here, that are aware of the bipartite nature of the data, are
accordingly called _bipartite_ estimators.

However, even if components of a bipartite estimator are multioutput
monopartite models, the final bipartite estimator will always be single-output
in the sense we previously defined (this should be further clarified in the next
section).

Regarding the `predict()` methods, a list of two sample sets must be provided,
similarly to what is expected by the `fit()` method of bipartite estimators.


The output of `bipartite_estimator.predict([X_test0, X_test1])`, however,
will be the *flattened* array of predictions to each instance combination,
*not* a predicted two-dimensional interaction matrix with shape
`(X_test0.shape[0], X_test1.shape[0])` as one might expect.

Although arguably unintuitive, we adopt this behaviour in order to facilitate
integration with `scikit-learn`'s scoring utilities, which always consider
bidimensional `y` arrays as multioutput targets.

Another detail to pinpoint is that some of the bipartite estimators provided
are actually able to receive concatenated sample pairs as input for `predict()`,
besides the general format we mentioned, of a list with two `X` sample sets.
This is the case of tree-based bipartite estimators in general and the
`GlobalSingleOutputWrapper` described in the following section. Such estimators
are consequently able to predict multiple specific interactions at a single call,
not subject to always computing predictions for all possible interactions between
`X_test0` and `X_test1`.

~simple but complete example

> Summary:
> 1. While the usual single-output monopartite estimators are trained on a sole
> `X_train` and a single-column `y_train`:
> ``monopartite_estimator.fit(X=X_train, y=y_train)``
> bipartite estimators receive two matrices `X_train0` and `X_train1`
> in a list, together with a `y_train` of shape
> `y_train.shape == (X_train0.shape[0], X_train1.shape[0])`:
> ``bipartite_estimator.fit(X=[X_train0, X_train1], y=y_train)``
> 2. The `predict()` method of bipartite estimators always returns a
> flattened array of predictions, to facilitate scoring.


## Validation
## Cross-validation and hyperparameter search


## Adapting monopartite estimators to bipartite datasets

There are two general ways of working with usual monopartite estimators when
dealing with bipartite data. Arguably the most natural is to build a new unified
`X` matrix whose rows are taken to be concatenations of a row from `X1` and
a row from `X2`. Accordingly, the `y` matrix is flattened with `y.reshape(-1, 1)`,
yielding a unidimensional column vector as expected by single-output
monopartite models. This procedure is defined by [1]_ as the
*global single-output* approach.

A `GlobalSingleOutputWrapper` is provided in this package to facilitate this
procedure.

Notice that considering all possible combinations of samples may be impeditive
in terms of memory usage or training time. Regarding memory issues, although
the transformed data is initially presented as references to avoid redundant
storage, some wrapped monopartite estimators will invariably copy the whole
dataset. For instance, `scikit-learn`'s tree-based models require the
training data to be contiguous in memory, and will copy them otherwise to
ensure that.

A common remedy to this problem is to subsample ...

The other general approach to adapt traditional models to bipartite data is
based on the idea of considering each sample domain as a separate task, so that
a multioutput monopartite estimator is fit to `X_train0` and `y_train`
(`y_train` being the full bidimensional interaction matrix), while another
receives `X_train1` and `y_train.T` (the transposed interaction matrix). 

Notice that the first estimator considers each column of `y_train` as a
different output to be predicted, and it does not have access to any extra
information about each of the columns (aside from the training targets),
that is, it does not consider the sample features at `X_train1`.

Analogously, the second estimator considers each row of `y_train` as a different
output, withou having access to the features describing each row
(kept by `X_train0`).

Since the first model (trained on `X_train0` and `y_train`) estimates new rows
for the interaction matrix, we thereafter call it a *rows estimator*.
Similarly, the second model (trained on `X_train1` and `y_train.T`) is intended
to predict new columns for the interaction matrix, so that it is referred to
as a *columns estimator*.

As estimators on each axis of the interaction matrix are completely
agnostic to the sample features on the other axis (they are "local" estimators),
this kind of strategy is called a *local multioutput* adaptation.

We hope it is now clear that the other adaptation method, the 
aforementioned *global single-output* approach, receives its name from the fact
that the
wrapped monopartite estimator expects to output a single value, and for that it
"globally" receives data from both sample domais at the same time (values from
`X_train0` and `X_train1` are used together in training).

However, notice that the local multioutput approach as described above is still
incapable of predicting interactions if both interacting intances are not present
in the training set. In order to circumvent this limitation, a second step
involving a second pair of multioutput monopartite estimators is introduced. 

The idea is that, after the described training of a rows estimator and a
columns estimator (now called *primary* rows/columns estimator), the models
are used to extend the interaction matrix to include the new instances on each
axis, and these newly predicted rows and columns are used to train a
*secondary* columns estimator and a *secondary* rows estimator, respectively.
Finally, the predictions of the secondary estimators are combined with an
arbitrary function to yield the final predictions. This function is commonly
chosen to be the simple average between them.

The following diagram illustrates the training procedure proposed by the
multi-output strategy. Notice how the initial `X_train0`, `X_train1` and
`y_train` can optionally be included to train the secondary estimators,
depending if the secondary estimators are able to take advantage of possible
inter-dependencies between its multiple outputs. If each output is treated
independently in any way, one can confidently use only the predictions of the
primary estimators to build the secondary models. 

While no reconstruction of `X` is needed in this approach, note that the
secondary estimators must be refit every time the wrapper's `predict()` is
called, increasing prediction time depending on the type of secondary
estimators chosen by the user.

We provide a `LocalMultioutputWrapper` class to easily implement this procedure.
Notice that compositions of single-output estimators can be used
instead of multipartite estimators, which can be easily implemented with 
`scikit-learn` meta-estimators such as `MultiOutputRegressor` and
`MultiOutputClassifier`.

{example}

> Summary:
> 1. The *global single-output* approach trains a single-output
> monopartite estimator on the flattened `y_train` and concatenated instance pairs 
> of a row from `X_train0` and a row from `X_train1`.
> 2. The *local multioutput* approach employs a composition of four multioutput
> monopartite estimators that treat rows and columns of `y_train` as different
> outputs to be predicted. Each has access only to `X_train0` or to `X_train1`,
> not being aware of the sample features on the other axis.


The `bipartite_learn.wrappers` module provides various tools 

Global meaning that 

the melter module

{native models, bipartite trees}

{CV and grid search}
