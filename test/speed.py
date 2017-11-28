# built-in imports
import timeit

# 3rd-party imports
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write

# local imports
from world import main


fs, x_int16 = wavread('test-mwm.wav')
x = x_int16 / (2 ** 15 - 1)
vocoder = main.World()

# profile
print(timeit.timeit("vocoder.encode(fs, x, f0_method='harvest')", globals=globals(), number=1))
