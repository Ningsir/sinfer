# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
# import numpy as np

# ext_modules = [
#     Extension("utils", ["lib/utils.pyx"],
#               extra_compile_args=["-fopenmp"],
#               extra_link_args=["-fopenmp"],
#               include_dirs=[np.get_include()])
# ]

# setup(
#     ext_modules=cythonize(ext_modules)
# )

import setuptools
from setuptools import setup, Extension
from torch.utils import cpp_extension
import numpy as np

compile_extra_args = ["-O3", "-fopenmp"]

setup(
    name='sinfer',
    packages=setuptools.find_packages(),
    ext_modules=[Extension("utils", ["src/utils.pyx"],
              extra_compile_args=compile_extra_args,
              extra_link_args=["-fopenmp"],
              include_dirs=[np.get_include()]),
              cpp_extension.CppExtension('cpp_core', ['src/gather.cpp'],
                                        extra_compile_args=compile_extra_args,
                                        extra_link_args=["-fopenmp", "-g"])],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    ext_package="sinfer"
    )
