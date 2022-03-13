# HyperTree
This code implements PBCTs based on its original proposal by Pliakos, Geurts and Vens in 2018<sup>1</sup>. Functionality will be extended to n-dimensional interaction tensors, where n instances of n different classes would interact or not for each database instance.

<sup>1</sup>Pliakos, Konstantinos, Pierre Geurts, and Celine Vens. "Global multi-output decision trees for interaction prediction." *Machine Learning* 107.8 (2018): 1257-1281.

## Installation
The package is available at PyPI and can be installed by the usual `pip` command:
```
$ pip install hypertree
```
Local installation can be done either by providing the `--user` flag to the above command or by cloning this repo and issuing `pip` afterwards, for example:
```
$ git clone https://github.com/pedroilidio/hypertree
$ cd PCT
$ pip install -e .
```
Where the `-e` option installs it as symbolic links to the local cloned repository, so that changes in it will reflect on the installation directly.
