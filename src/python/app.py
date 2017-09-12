# built-in imports
import sys

# 3rd-party imports
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write
# local imports
#for now, let's stay pysig independent, since we may want to distribute our code freely later
#sys.path.append('/Users/tuandinh/bakup/pysig/src/python')
#from pysig import track


#sys.path.append('world')


#import pyximport; pyximport.install()


from world.main import World
#import pyworld as pw # official

if __name__ == '__main__':   
    name='test-mwm'
    fs, x_int16 = wavread('{}.wav'.format(name))
    x = x_int16 / (2 ** 15 - 1)
    vocoder = World()
    # time, value = vocoder.get_f0(x, fs)
    # analysis
    dat = vocoder.encode(fs, x, f0_method='harvest')
    if 0:
        # global pitch scaling
        dat = vocoder.scale_pitch(dat, 2)
    if 0:
        # global duration scaling
        dat = vocoder.scale_duration(dat, 2)
    if 0:
        dat = vocoder.warp_spectrum(dat, 0.8)
    # synthesis
    dat = vocoder.decode(dat)
    # play
    import simpleaudio as sa
    snd = sa.play_buffer((dat['out'] * 2 ** 15).astype(np.int16), 1, 2, fs)
    snd.wait_done()
    if 0:
        # draw smt
        vocoder.draw(x, dat)