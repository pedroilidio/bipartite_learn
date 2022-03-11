from setuptools import setup, Extension
from pathlib import Path
from Cython.Build import cythonize
import numpy

PATH_HERE = Path(__file__).parent
README = (PATH_HERE/"README.md").read_text()

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
    version='0.0.1b1',
    description='HyperTrees in Python.',
    include_dirs=[numpy.get_include()],
    long_description=README,
    long_description_content_type="text/markdown",
    url='http://github.com/pedroilidio/hypertree',
    author='Pedro Ilídio',
    author_email='pedrilidio@gmail.com',
    license='GPLv3',
    packages=['hypertree'],
    # scripts=['bin/hypertree'],
    zip_safe=False,
    install_requires=['sklearn', 'numpy', 'cython'],

    ext_modules=cythonize(
        extensions,
        language_level="3",
    ),
)
