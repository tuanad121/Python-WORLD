from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["world/synthesis.py", 'world/d4c.py', 'world/cheaptrick.py'])
)
