import sys
import numpy as np
sys.path.append('/Users/tuandinh/bakup/pysig/src/python')
from pysig import track
from dio import ZeroCrossingEngine # yours
from dio import GetF0Candidates
#import pyworld # official


SHORT_MAX = 32767 #normalize input signal
name='test/test-mwm'
inp_wav = track.Wave.wavread('%s.wav' % name, channel=0)


#result =world.dio(np.ndarray[double, ndim=1, mode="c"] x not None, int fs, double period, option):

#print(inp_wav.get_value())
#results=dio.Dio(inp_wav.get_value()/SHORT_MAX, inp_wav.fs)
frame_period=5
temporal_positions = np.arange(0, np.size(inp_wav.get_value())/inp_wav.fs, frame_period/1000)
#print(temporal_positions)
negative_zero_cross = ZeroCrossingEngine(inp_wav.get_value()/SHORT_MAX, inp_wav.fs);
positive_zero_cross = ZeroCrossingEngine(-inp_wav.get_value()/SHORT_MAX, inp_wav.fs);
peak = ZeroCrossingEngine(np.diff(inp_wav.get_value()/SHORT_MAX), inp_wav.fs);
dip = ZeroCrossingEngine(-np.diff(inp_wav.get_value()/SHORT_MAX), inp_wav.fs);
f0_candidates, f0_deviations = GetF0Candidates(negative_zero_cross, positive_zero_cross, peak, dip, temporal_positions)
print(f0_deviations)
print('done')
#print(results['temporal_positions'])

