from setuptools import setup
from Cython.Build import cythonize

if 0:  # for now, don't cythonize
    setup(
        ext_modules=cythonize(["world/dio.py", "world/synthesis.py", 'world/d4c.py', 'world/cheaptrick.py', 'world/harvest.py'])
    )
