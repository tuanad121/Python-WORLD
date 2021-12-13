from setuptools import setup, find_packages

setup(
    name='python-world',
    version='0.0.1',
    description='Line-by-line implementation of the WORLD vocoder (Matlab, C++) in python.',
    url='https://github.com/tuanad121/Python-WORLD',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'numba',
    ]
)