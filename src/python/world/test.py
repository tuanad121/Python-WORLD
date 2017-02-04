# TODO: use main.py wrapper to do everything

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
import unittest

# local imports
#for now, let's stay pysig independent, since we may want to distribute our code freely later
#sys.path.append('/Users/tuandinh/bakup/pysig/src/python')

#from pysig import track
from dio import dio
from stonemask import stonemask
from havest import havest
from cheaptrick import cheaptrick
from d4c import d4c
from synthesis import synthesis
#import pyworld as pw # official
from pymatbridge import Matlab # pip install pymatbridge
from numpy import genfromtxt

# C version calling
#pyDioOpt = pw.pyDioOption()
#_f0, t = pw.dio(x, fs, pyDioOpt)    # raw pitch extractor
#f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
#sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
#ap = pw.d4c(x, f0, t, fs)         # extract aperiodicity
#y = pw.synthesize(f0, sp, ap, fs, pyDioOpt.option['frame_period'])

class Test(unittest.TestCase):

    def setUp(self):
        path = '/Users/tuandinh/my desk/working/duration-conversion/dat/Harvard/template/'
        self.wav_list = ['{}CM8SC{:02}.wav'.format(path, elm) for elm in range(10, 26)]

    def test_F0(self):
        for elm in self.wav_list:
            # extract f0 by python
            fs, x_int16 = wavread(elm)
            x = x_int16 / (2 ** 15 - 1)
            #f0_data = dio(x, fs)
            #f0_data['f0'] = stonemask(x, fs, f0_data['temporal_positions'], f0_data['f0'])
            f0_data = havest(x, fs)
            # extract f0 by matlab
            f0_matlab, _ = call_matlab(elm)
            assert len(f0_data['f0']) == len(f0_matlab), print('{} and {}'.format(len(f0_data['f0']), len(f0_matlab)))
            np.testing.assert_array_almost_equal(f0_data['f0'], f0_matlab, decimal=2)

    # def test_spectrum(self):
    #     for elm in self.wav_list:
    #         # extract f0, spectrogram by python
    #         fs, x_int16 = wavread(elm)
    #         x = x_int16 / (2 ** 15 - 1)
    #         f0_data = havest(x, fs, frame_period=1)
    #         filter_object = cheaptrick(x, fs, f0_data)
    #
    #         # extract f0, spectrogram by matlab
    #         f0_matlab, spec_matlab = call_matlab(elm)
    #         np.testing.assert_array_almost_equal(filter_object['spectrogram'], spec_matlab, decimal=2)

    #def test_wav(self):

def call_matlab(filename):
    mlab = Matlab(executable='/Applications/MATLAB_R2016b.app/bin/matlab')
    mlab.start()
    matlab_code = \
        '''
        addpath('../../matlab/world-0.2.1_matlab');
        [x, fs] = audioread('{}');
        if 0 % You can use Dio
        f0_parameter = Dio(x, fs);
        % StoneMask is an option for improving the F0 estimation performance.
        % You can skip this processing.
        f0_parameter.f0 = StoneMask(x, fs,...
          f0_parameter.temporal_positions, f0_parameter.f0);
        else
        f0_parameter = Harvest(x, fs);
        end;
        dlmwrite('f0.csv',f0_parameter.f0);

        spectrum_parameter = CheapTrick(x, fs, f0_parameter);
        dlmwrite('spec.csv', spectrum_parameter.spectrogram);
        %source_parameter = D4C(x, fs, f0_parameter);

        %y = Synthesis(source_parameter, spectrum_parameter);

        '''.format(filename)
    mlab.run_code(matlab_code)
    mlab.stop()

    f0 = genfromtxt('f0.csv', delimiter=',')
    spec = genfromtxt('spec.csv', delimiter=',')
    from os import remove
    remove('f0.csv')
    remove('spec.csv')
    return f0, spec
if __name__ == '__main__':
    unittest.main()