from pathlib import Path

import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy import signal

from world import main

wav_path = Path('../test/test-mwm.wav')
print(wav_path)
fs, x_int16 = wavread(wav_path)
x = x_int16 / (2 ** 15 - 1)

if 0:  # resample
    fs_new = 16000
    x = signal.resample_poly(x, fs_new, fs)
    fs = fs_new

if 0:  # low-cut
    B = signal.firwin(127, [0.01], pass_zero=False)
    A = np.array([1.0])
    if 0:
        import matplotlib.pyplot as plt
        w, H = signal.freqz(B, A)

        fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 6))
        ax1.plot(w / np.pi, abs(H))
        ax1.set_ylabel('magnitude')
        ax2.plot(w / np.pi, np.unwrap(np.angle(H)))
        ax2.set_ylabel('unwrapped phase')
        plt.show()
    x = signal.lfilter(B, A, x)

vocoder = main.World()

# analysis
dat = vocoder.encode(fs, x, f0_method='harvest')
if 0:  # global pitch scaling
    dat = vocoder.scale_pitch(dat, 1.5)
if 0:  # global duration scaling
    dat = vocoder.scale_duration(dat, 2)
if 0:  # fine-grained duration modification
    vocoder.modify_duration(dat, [1, 1.5], [0, 1, 3, -1])  # TODO: look into this


# dat['f0'] = np.r_[np.zeros(5), dat['f0'][:-5]]

# synthesis
dat = vocoder.decode(dat)
if 0:  # audio
    import simpleaudio as sa
    snd = sa.play_buffer((dat['out'] * 2 ** 15).astype(np.int16), 1, 2, fs)
    snd.wait_done()
if 0:  # visualize
    vocoder.draw(x, dat)

wavwrite(wav_path.with_name(wav_path.stem + '-resynth.wav'), fs, (dat['out'] * 2 ** 15).astype(np.int16))