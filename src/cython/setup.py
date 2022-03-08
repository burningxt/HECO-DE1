from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize((Extension("slow_loops_cy",
                                     sources=["slow_loops_cy.pyx"],
                                     include_dirs=[np.get_include()], ),),
                          compiler_directives={'language_level': "3"},
                          annotate=True)
)

