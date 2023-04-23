<p align="center">
  <img src="https://github.com/pedroilidio/bipartite_learn/raw/main/docs/_static/logos/bipartite_learn_wide_banner.png" width="100%">
</p>

[![PyPI version](https://badge.fury.io/py/bipartite_learn.svg)](https://badge.fury.io/py/bipartite_learn)
[![Documentation Status](https://readthedocs.org/projects/bipartite-learn/badge/?version=latest)](https://bipartite-learn.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

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

A documentation for `bipartite_learn`, still in its infancy, can be found at
[bipartite-learn.rtfd.io](https://bipartite-learn.readthedocs.io).
Please refer to the [User Guide](https://bipartite-learn.readthedocs.io/en/latest/user_guide.html) for more information
on how to use this package.

## Installation
`bipartite_learn` is available on PyPI, and thus can be installed with `pip`:
```
$ pip install bipartite_learn
```
Installation from source can be done by cloning this repository and calling
`pip install` on the downloaded folder.
```
$ git clone https://github.com/pedroilidio/bipartite_learn
$ cd bipartite_learn
$ pip install --editable .
```
The optional `--editable` (or `-e`) flag links the installed package to the
local cloned repository, so that local changes in it will immediatly be active
without the need for reinstallation.

Currently, `bipartite_learn` uses the 1.3 version of `scikit-learn`, not yet
released. As such, after installing with `pip` one may have to mannually download this version from github:

```
$ pip install git+https://github.com/scikit-learn/scikit-learn@893d5accaf9d16f447645e704f85a216187564f7#egg=scikit-learn
```