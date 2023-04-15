<p align="center">
  <img src="https://github.com/pedroilidio/bipartite_learn/raw/main/docs/logos/bipartite_learn_wide_banner.png" width="100%">
</p>

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