#! /usr/bin/env python
# Copyright (C) 2022 Pedro Ilídio <ilidio@alumni.usp.br>
# License: 3-clause BSD

from setuptools import setup, Extension, find_packages
from pathlib import Path
from Cython.Build import cythonize
import numpy

PATH_ROOT = Path(__file__).parent
README = (PATH_ROOT/"README.md").read_text()
# Get __version__ from _version.py
exec((PATH_ROOT/"hypertree/_version.py").read_text())
VERSION = __version__

extensions = [
    Extension(
        "hypertree.tree._nd_tree", ["hypertree/tree/_nd_tree.pyx"]),
    Extension(
        "hypertree.tree._nd_splitter", ["hypertree/tree/_nd_splitter.pyx"]),
    Extension(
        "hypertree.tree._nd_criterion", ["hypertree/tree/_nd_criterion.pyx"]),
]

setup(
    name='hypertree',
    version=VERSION,
    description='HyperTrees in Python.',
    include_dirs=[numpy.get_include()],
    long_description=README,
    long_description_content_type="text/markdown",
    url='http://github.com/pedroilidio/hypertree',
    author='Pedro Ilídio',
    author_email='pedrilidio@gmail.com',
    license='new BSD',
    packages=find_packages(),
    # scripts=['bin/hypertree'],
    zip_safe=False,

    install_requires=[
        'cython>=0.29.27',
        'sklearn>=1.0.2',
        'numpy>=1.22.2',
        'imbalanced-learn>=0.7.0',
    ],
    ext_modules=cythonize(
        extensions,
        language_level="3",
    ),
)
