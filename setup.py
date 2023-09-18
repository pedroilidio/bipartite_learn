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
exec((PATH_ROOT/"bipartite_learn/_version.py").read_text())
VERSION = __version__

extensions = [Extension("bipartite_learn.tree.*", ["bipartite_learn/tree/*.pyx"])]

setup(
    name='bipartite_learn',
    version=VERSION,
    description='Machine learning estimators for bipartite data.',
    include_dirs=[numpy.get_include()],
    long_description=README,
    long_description_content_type="text/markdown",
    url='http://github.com/pedroilidio/bipartite_learn',
    author='Pedro Ilídio',
    author_email='pedrilidio@gmail.com',
    license='new BSD',
    packages=find_packages(),
    zip_safe=False,

    install_requires=[
        'cython==0.29.33',
        'scikit-learn==1.3.0',
        'numpy>=1.22.2',
        'imbalanced-learn==0.9.1',
    ],
    extras_require={
        'docs': ['sphinx', 'pydata-sphinx-theme'],
    },

    ext_modules=cythonize(
        extensions,
        language_level="3",
        gdb_debug=True,
    ),
)
