from setuptools import setup
from Cython.Build import cythonize
import numpy


setup(
    name='BSCSimulator',
    ext_modules=cythonize("BSCSimulator/mincostflow.pyx",
                          language_level="3"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
