# TODO: write this as a unit test using UnitTest(), using a list of waveforms (or just one)
# TODO: call oct2py from here
# TODO: Use self.assertClose for F0

# built-in imports
import sys

# 3rd-party imports
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write
from matplotlib import pyplot
#import unittest

# local imports
#for now, let's stay pysig independent, since we may want to distribute our code freely later
#sys.path.append('/Users/tuandinh/bakup/pysig/src/python')

#from pysig import track
from dio import dio
from dio import ZeroCrossingEngine # yours
from dio import GetF0Candidates
from stoneMask import StoneMask
from cheapTrick import CheapTrick
from d4c import D4C
from synthesis import Synthesis
import pyworld as pw # official
from call_matlab import call_matlab

# C version calling
#pyDioOpt = pw.pyDioOption()
#_f0, t = pw.dio(x, fs, pyDioOpt)    # raw pitch extractor
#f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
#sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
#ap = pw.d4c(x, f0, t, fs)         # extract aperiodicity
#y = pw.synthesize(f0, sp, ap, fs, pyDioOpt.option['frame_period'])

# class Test(unittest.TestCase):
#     def setUp(self):
#         path = '/Users/tuandinh/my desk/working/duration-conversion/dat/Harvard/template/'
#         self.wav_list = ['{}CM8SC{:02}.wav'.format(path, elm) for elm in range(10, 26)]
#
#     def test_F0(self):
#         for elm in self.wav_list:
#             # extract f0 by python
#             fs, x_int16 = wavread(elm)
#             x = x_int16 / (2 ** 15 - 1)
#             f0_data = dio(x, fs, frame_period=5)
#             #no_stonemask = np.copy(f0_data['f0'])
#             #f0_data['f0'] = StoneMask(x, fs,f0_data['temporal_positions'], f0_data['f0'])
#             #filter_object = CheapTrick(x, fs, f0_data)
#             #source_object = D4C(x, fs, f0_data)
#             # extract f0 by matlab
#             f0_matlab, _ = call_matlab(elm)
#
#             np.testing.assert_array_almost_equal(f0_data['f0'], f0_matlab, decimal=5)
#
#     def test_spectrum(self):
#         pass


if __name__ == '__main__':
    #unittest.main()
    from matplotlib import pyplot as plt
    path = '/Users/tuandinh/my desk/working/duration-conversion/dat/Harvard/template/'
    wav_list = ['{}CM8SC{:02}.wav'.format(path, elm) for elm in range(10, 11)] 
    wav_list = ['test/test-mwm.wav']
    for elm in wav_list:
        # extract f0 by python
        fs, x_int16 = wavread(elm)
        x = x_int16 / (2 ** 15 - 1)
        f0_data = dio(x, fs, frame_period=5)
        f0_data['f0'] = StoneMask(x, fs,f0_data['temporal_positions'], f0_data['f0'])            
        #filter_object = CheapTrick(x, fs, f0_data)
        #source_object = D4C(x, fs, f0_data)
        # extract f0 by matlab
        f0_matlab, _ = call_matlab(elm)
        #fig, ax = plt.subplot()
        #assert len(f0_data['f0']) == len(f0_matlab)
        #plt.plot(f0_matlab)
        f, ax = plt.subplots(2, sharex=True)
        ax[0].plot(f0_data['f0'])
        ax[1].plot(f0_matlab)
        plt.show()
    