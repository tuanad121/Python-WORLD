# built-in imports
import sys

# 3rd-party imports
import numpy as np

from scipy.io.wavfile import read as wavread
from matplotlib import pyplot
# local imports
#for now, let's stay pysig independent, since we may want to distribute our code freely later
#sys.path.append('/Users/tuandinh/bakup/pysig/src/python')
#from pysig import track
from dio import Dio
from dio import ZeroCrossingEngine # yours
from dio import GetF0Candidates
#import pyworld as pw # official

name='test/test-mwm'
fs, x_int16 = wavread('{}.wav'.format(name))
x = x_int16 / (2 ** 15 - 1)

assert(all(isinstance(elm, np.float) for elm in x))
f0_data = Dio(x,fs)
#pyDioOpt = pw.pyDioOption()
#f02, t = pw.dio(x, fs, pyDioOpt)

f0_matlab = np.genfromtxt('test/dat_mat.csv', delimiter = ',')
fig, ax = pyplot.subplots()
ax.plot(f0_data['f0'], label = 'F0_DIO_python')
ax.plot(f0_matlab,'g', label = 'F0_DIO_matlab')
#ax.plot(np.abs(f0_matlab - f0_data['f0']), 'r', label = 'different')
ax.legend(loc = 1)
pyplot.show()
print('done')

