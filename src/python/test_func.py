import pyworld as pw
# from scipy.io.wavfile import read as wavread
#
# import timeit
# name='test-mwm'
# fs, x_int16 = wavread('{}.wav'.format(name))
# x = x_int16 / (2 ** 15 - 1)
#
# print(timeit.timeit("pw.harvest(x,fs)", globals=globals(), number=1))
import numpy as np
a = np.arange(20)
a = np.reshape(a, [4,5])
print(a)