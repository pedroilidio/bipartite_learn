from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

DIR_HERE = Path(__file__).parent

extensions = [
    Extension("patched_modules.*", [DIR_HERE/"*.pyx"],
        extra_compile_args=["-g"],  # maybe unnecessary.
        extra_link_args=["-g"],
        # define_macros=[  # Causes error.
        #     ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
        # ],
    )
]

setup(
    name='sklearn treeeee',
    ext_modules=cythonize(
        extensions,
        gdb_debug=True,
        annotate=True,
        # language="c++",
        language_level="3",
    ),
    include_dirs=[numpy.get_include()],
)
