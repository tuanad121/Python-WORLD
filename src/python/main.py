# built-in imports
import sys

# 3rd-party imports
import numpy as np
from scipy.io.wavfile import read as wavread

# local imports
#for now, let's stay pysig independent, since we may want to distribute our code freely later
#sys.path.append('/Users/tuandinh/bakup/pysig/src/python')
#from pysig import track
from dio import ZeroCrossingEngine # yours
from dio import GetF0Candidates
#import pyworld # official


name='test/test-mwm'
fs, x_int16 = wavread('{}.wav'.format(name))
x = x_int16 / (2 ** 15)
assert isinstance(x, np.float)

#result =world.dio(np.ndarray[double, ndim=1, mode="c"] x not None, int fs, double period, option):

#print(inp_wav.get_value())
#results=dio.Dio(inp_wav.get_value()/SHORT_MAX, inp_wav.fs)
frame_period = 5
temporal_positions = np.arange(0, len(x) / fs, frame_period / 1000)
#print(temporal_positions)
negative_zero_cross = ZeroCrossingEngine( x, fs);
positive_zero_cross = ZeroCrossingEngine(-x, fs);
peak = ZeroCrossingEngine(np.diff(x), fs);
dip = ZeroCrossingEngine(-np.diff(x), fs);
f0_candidates, f0_deviations = GetF0Candidates(negative_zero_cross, positive_zero_cross, peak, dip, temporal_positions)
print(f0_deviations)
print('done')
#print(results['temporal_positions'])

