from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["world/Synthesis.py", 'world/D4C.py', 'world/cheapTrick.py'])
)