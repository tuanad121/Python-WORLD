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
from StoneMask import StoneMask
from cheapTrick import CheapTrick
from D4C import D4C
#import pyworld as pw # official

name='test/test-mwm'
fs, x_int16 = wavread('{}.wav'.format(name))
x = x_int16 / (2 ** 15 - 1)

assert(all(isinstance(elm, np.float) for elm in x))
f0_data = Dio(x,fs)
#no_stonemask = np.copy(f0_data['f0'])
f0_data['f0'] = StoneMask(x, fs,f0_data['temporal_positions'], f0_data['f0'])

#print(f0_data['f0'])5

#py wrapper testing
#pyDioOpt = pw.pyDioOption()
#f02, t = pw.dio(x, fs, pyDioOpt)

#f0_matlab = np.genfromtxt('dat_mat.csv', delimiter = ',')
#f0_data['f0'] = f0_matlab
#fig, ax = pyplot.subplots()
#ax.plot(f0_matlab,'g', label = 'F0_DIO_matlab')
#ax.plot(f0_data['f0'], label = 'F0_python')
#ax.plot(np.abs(f0_data['f0'] - f0_matlab),'r', label = 'DIFF')
#ax.legend(loc = 0)

#spectrum_parameter = CheapTrick(x, fs, f0_data)
source_parameter = D4C(x, fs, f0_data)
#pyplot.imshow(spectrum_parameter['spectrogram'])
#pyplot.show()
print('done')

