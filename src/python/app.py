# built-in imports
import timeit

# 3rd-party imports
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write

# local imports
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

levinson = Levinson().compute  # functional form for convenience

if __name__ == '__main__':   
    name = 'test-mwm'
    fs, x_int16 = wavread('{}.wav'.format(name))
    x = x_int16 / (2 ** 15 - 1)
    vocoder = main.World()
    dat = vocoder.encode(fs, x, f0_method='harvest')
    if 0:
        # global pitch scaling
        dat = vocoder.scale_pitch(dat, 2)  # be careful when scaling the pitch down too much
    if 0:
        # global duration scaling
        dat = vocoder.scale_duration(dat, 2)
    if 0:
        dat = vocoder.warp_spectrum(dat, 1.2)
    if 0:  # downsampling example
        dat['spectrogram'][257:, :] = 1e-12
    if 0:  # cepstral smoothing
        dat = vocoder.to_cepstrum(dat)
        cep = dat['cepstrum']
        D, N = cep.shape
        # liftering
        L = 30
        cep = cep[:L,:]
        #cep = cep[:40,:]
        #cep = np.r_[cep, np.zeros((((dat['spectrogram'].shape[0] - 1) * 2 - cep.shape[0]), cep.shape[1]))]
        # process cep here
        pass
        # reconstruct now
        cep2 = np.zeros((D, N))
        cep2[:L, :] = cep
        dat['cepstrum'] = cep2
        dat = vocoder.from_cepstrum(dat)
    if 1:  # LPC smoothing
        from scipy.fftpack import ifft
        from scipy.signal import freqz
        # Levinson Durbin on spectrum
        spec = dat['spectrogram']
        H = np.zeros_like(spec)
        D, N = spec.shape
        for i in range(N):
            sp = spec[:, i]
            r = ifft(sp ** 2, n=(D-1) * 2).real
            r = r[:D]
            rms1 = np.sqrt(np.mean(sp**2))
            a, _, _ = levinson(r, order=30)
            w, h = freqz([1.0], a, worN=513)
            h = h.real

            rms2 = np.sqrt(np.mean(h.real**2))
            g = rms1 / rms2
            H[:, i] = g * h
        dat['spectrogram'] = H

    # synthesis
    dat = vocoder.decode(dat)
    import simpleaudio as sa
    snd = sa.play_buffer((dat['out'] * 2 ** 15).astype(np.int16), 1, 2, fs)
    snd.wait_done()
    if 1:
        # visualize
        vocoder.draw(x, dat)
