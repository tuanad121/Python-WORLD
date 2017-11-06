# built-in imports
import timeit

# 3rd-party imports
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write

# local imports
from world import main


if __name__ == '__main__':   
    name = 'test-mwm'
    fs, x_int16 = wavread('{}.wav'.format(name))
    x = x_int16 / (2 ** 15 - 1)
    vocoder = main.World()
    # profile
    print(timeit.timeit("vocoder.encode(fs, x, f0_method='harvest')", globals=globals(), number=1))
