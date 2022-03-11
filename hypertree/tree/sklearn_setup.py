import os

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("tree", parent_package, top_path)
    libraries = []
    if os.name == "posix":
        libraries.append("m")
    config.add_extension(
        "_nd_tree",
        sources=["_nd_tree.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        language="c++",
        extra_compile_args=["-O3"],
    )
    config.add_extension(
        "_nd_splitter",
        sources=["_nd_splitter.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    )
    config.add_extension(
        "_nd_criterion",
        sources=["_nd_criterion.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    )
    #config.add_subpackage("tests")

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())
