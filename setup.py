import setuptools
from setuptools import setup, Extension
from torch.utils import cpp_extension
import numpy as np
import os
import sysconfig

current_dir = os.path.abspath(os.path.dirname(__file__))

python_include = sysconfig.get_paths()["include"]
# 指定spdlog库的路径
spdlog_include_dir = os.path.join(python_include, "../")
spdlog_lib_dir = os.path.join(python_include, "../../lib")

compile_extra_args = ["-std=c++14", "-O3", "-lpthread", "-fopenmp"]
# compile_extra_args = ["-std=c++14", "-lpthread", "-fopenmp", "-g"]
link_args = ["-fopenmp", "-g"]

setup(
    name="sinfer",
    packages=setuptools.find_packages(),
    ext_modules=[
        Extension(
            "utils",
            ["src/cython/utils.pyx"],
            extra_compile_args=compile_extra_args,
            extra_link_args=link_args,
            include_dirs=[np.get_include()],
        ),
        cpp_extension.CppExtension(
            "cpp_core",
            ["src/cpp/gather.cpp", "src/cpp/free.cpp", "src/cpp/api.cpp", "src/cpp/utils.cpp"],
            libraries=["spdlog"],
            extra_compile_args=compile_extra_args,
            extra_link_args=link_args,
            include_dirs=[os.path.join(current_dir, "include"), spdlog_include_dir],
            library_dirs=[spdlog_lib_dir],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    ext_package="sinfer",
)
