from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("tests.utils.tree_utils", ["tests/utils/tree_utils.pyx"],
        extra_compile_args=["-g"],
        extra_link_args=["-g"],
    ),
]

setup(
    name='bipartite_learn_tests',
    ext_modules=cythonize(
        extensions,
        gdb_debug=True,
        annotate=True,
        language_level="3",
    ),
    include_dirs=[numpy.get_include()],
)
