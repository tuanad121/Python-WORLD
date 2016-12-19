# built-in imports
import sys

# 3rd-party imports
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write
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
from Synthesis import Synthesis
import pyworld as pw # official


name='test/test-mwm'
fs, x_int16 = wavread('{}.wav'.format(name))
x = x_int16 / (2 ** 15 - 1)

assert(all(isinstance(elm, np.float) for elm in x))
f0_data = Dio(x,fs)
#no_stonemask = np.copy(f0_data['f0'])
f0_data['f0'] = StoneMask(x, fs,f0_data['temporal_positions'], f0_data['f0'])

#print(f0_data['f0'])5

#py wrapper testing
pyDioOpt = pw.pyDioOption()
_f0, t = pw.dio(x, fs, pyDioOpt)    # raw pitch extractor
f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
ap = pw.d4c(x, f0, t, fs)         # extract aperiodicity
#y = pw.synthesize(f0, sp, ap, fs, pyDioOpt.option['frame_period'])

#f0_matlab = np.genfromtxt('dat_mat.csv', delimiter = ',')
#f0_data['f0'] = f0_matlab
#fig, ax = pyplot.subplots()
#ax.plot(f0_matlab,'g', label = 'F0_DIO_matlab')
#ax.plot(f0_data['f0'], label = 'F0_python')
#ax.plot(np.abs(f0_data['f0'] - f0_matlab),'r', label = 'DIFF')
#ax.legend(loc = 0)

filter_object = CheapTrick(x, fs, f0_data)
source_object = D4C(x, fs, f0_data)
sp2 = (filter_object['spectrogram'].T).copy(order='C')# still different from C spectrogram, don't know why
ap2 = (source_object['aperiodicity'].T).copy(order='C')

y2 = pw.synthesize(source_object['f0'], sp2,\
                   ap, fs, pyDioOpt.option['frame_period'])

#y = Synthesis(source_object, filter_object)
write('test.wav', fs, y2)
#write('test2.wav', fs, y)
fig, ax = pyplot.subplots(nrows=2, figsize=(12,8), sharex=True)
ax[0].imshow(20 * np.log10(sp[705: 720,:]).T)
ax[0].set_title('C spectrogram')
ax[1].imshow(20 * np.log10(sp2[705: 720,:]).T)
ax[1].set_title('python')
#pyplot.imshow(spectrum_parameter['spectrogram'])
#pyplot.show()
#pyplot.plot(y)
pyplot.show()
print('done')

