import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy.fftpack import ifft
from scipy.signal import freqz

from world import main


class Levinson(object):
    # implemented in a class, since it may be called many times,
    # but we don't want to allocate working memory each time
    def __init__(self, max_order=255):
        self.max_order = max_order
        self.a = np.ones(max_order + 1, dtype=np.float64)
        self.tmp = np.ones(max_order, dtype=np.float64)
        self.k = np.zeros(max_order, dtype=np.float64)

    def compute(self, R: np.ndarray, order: int):
        """e is the prediction error of the specified order"""
        assert R.ndim == 1, "input must be one-dimensional"
        assert order < R.shape[0], "order too large"
        assert order <= self.max_order
        assert np.isreal(R).all()
        a = self.a
        tmp = self.tmp
        k = self.k
        e = R[0]
        for i in range(1,order+1):
            klast = a[i] = k[i-1] = (R[i] - sum(a[1:i] * R[i-1:0:-1])) / e  # for i==0 the sum is zero
            tmp[:order+1] = a[:order+1]      # must be a copy, not a view...
            a[1:i] -= klast * tmp[i-1:0:-1]  # ...since variable "a" would otherwise refer to itself as it is changing
            e *= 1.0 - klast * klast
        a[1:order+1] = -a[1:order+1]  # since we want the output as standard filter coefficients
        return a[:order+1], e, k[:order]


name = '../test/test-mwm'
fs, x_int16 = wavread(f'{name}.wav')
x = x_int16 / (2 ** 15 - 1)
vocoder = main.World()
dat = vocoder.encode(fs, x, f0_method='harvest')

emphasis = True
if emphasis:
    D, N = dat['spectrogram'].shape
    w, h = freqz([1, -.95], [1], D)
    dat['spectrogram'] = (dat['spectrogram'].T * np.abs(h)).T

warp = 2
if warp:
    vocoder.warp_spectrum(dat, warp)  # TODO: perhaps upsample also

if 1:  # LPC modeling via Levinson-Durbin on spectrum
    order = 24
    levinson = Levinson()
    spec = dat['spectrogram']
    H = np.zeros_like(spec)
    D, N = spec.shape

    A = np.empty((order + 1, N))
    E = np.empty(N)
    spec2 = spec ** 2
    R = ifft(spec2.T, n=(D - 1) * 2).T.real[:D, :]  # calculate auto-correlation
    for i in range(N):
        r = R[:, i]
        A[:, i], _, _ = levinson.compute(r, order)
        E[i] = (2 * r[0]) ** 0.5
        # E[i] = np.mean(spec[:, i] ** 2) ** 0.5  # just a reminder of the relationship
    # process A (and E) here
    for i in range(N):
        _, h = freqz([1.0], A[:, i], worN=D)  # TODO: non-uniformly sample if unwarping is required
        h = np.abs(h)
        h *= E[i] / np.mean(h ** 2) ** 0.5  # apply gain
        H[:, i] = h
        if 0:  # see spectra
            from matplotlib import pyplot as plt
            plt.plot((spec[:, i]))
            plt.plot((H[:, i]))
            plt.title(str(i))
            plt.show()
    dat['spectrogram'] = H
    if 1:  # see spectrograms
        from matplotlib import pyplot as plt
        plt.subplot(211)
        plt.pcolor(np.log(spec))
        plt.subplot(212)
        plt.pcolor(np.log(H))
        plt.show()

if warp:  # unwarp
    vocoder.warp_spectrum(dat, 1 / warp)

if emphasis:  # de-emphasis
    D, N = dat['spectrogram'].shape
    w, h = freqz([1], [1, -.95], D)
    dat['spectrogram'] = (dat['spectrogram'].T * np.abs(h)).T


# synthesis
dat = vocoder.decode(dat)
import simpleaudio as sa
out = (dat['out'] * 2 ** 15).astype(np.int16)
snd = sa.play_buffer(out, 1, 2, fs)
wavwrite(f'{name}-synth.wav', fs, out)
if 0:  # visualize
    vocoder.draw(x, dat)
snd.wait_done()
