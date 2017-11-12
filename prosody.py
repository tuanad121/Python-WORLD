import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write

from world import main


fs, x_int16 = wavread('test-mwm.wav')
x = x_int16 / (2 ** 15 - 1)

vocoder = main.World()

# analysis
dat = vocoder.encode(fs, x, f0_method='harvest')
if 1:  # global pitch scaling
    dat = vocoder.scale_pitch(dat, 0.5)
if 0:  # global duration scaling
    dat = vocoder.scale_duration(dat, 2)
if 1:  # fine-grained duration modification
    vocoder.modify_duration(dat, [1, 1.5], [0, 1, 3, -1])

# synthesis
dat = vocoder.decode(dat)
if 1:  # audio
    import simpleaudio as sa
    snd = sa.play_buffer((dat['out'] * 2 ** 15).astype(np.int16), 1, 2, fs)
    snd.wait_done()
if 0:  # visualize
    vocoder.draw(x, dat)
