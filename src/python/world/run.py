from world.havest import havest
from dio import dio
from cheaptrick import cheaptrick
from stonemask import stonemask
from d4c import d4c
from synthesis import synthesis

# 3rd party imports
import numpy as np
from numpy import genfromtxt
from scipy.io.wavfile import write
from scipy.io.wavfile import read as wavread

fs, x_int16 = wavread('test-mwm.wav')
x = x_int16 / (2 ** 15 - 1)
#x = x[3*fs:4*fs ]
source = dio(x, fs)
source['f0'] = stonemask(x, fs, source['temporal_positions'], source['f0'])
#source = havest(x, fs)
filter = cheaptrick(x, fs, source)
source = d4c(x, fs, source)
# for debug synthesis function

y = synthesis(source, filter)
import simpleaudio as sa
snd = sa.play_buffer((y * 2**15).astype(np.int16), 1, 2, fs)
snd.wait_done()
write('test-mwm-re.wav', fs, y)
