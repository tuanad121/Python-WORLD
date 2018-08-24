import numpy as np
import numba
from scipy.fftpack import fft, ifft, fftshift
from scipy.signal import hanning

import random

def get_seeds_signals(fs: int, fft_size: int=None, noise_length: int=None):
    if fft_size == None:
        fft_size = int(1024 * (2 ** np.ceil(np.log2(fs / 48000))))
    if noise_length == None:
        noise_length = int(2 ** np.ceil(np.log2(fs / 2)))
    w = np.arange(fft_size // 2 + 1) * fs / fft_size
    frequency_interval = 3000
    frequency_range = frequency_interval * 2
    upper_limit = 15000
    number_of_aperiodicities = int(2 + np.floor(min(upper_limit, fs / 2 - frequency_interval) / frequency_interval))

    pulse = np.zeros((fft_size, number_of_aperiodicities))
    noise = np.zeros((noise_length, number_of_aperiodicities))

    modified_velvet_noise = generate_modified_velvet_noise(noise_length, fs)
    spec_n = fft(modified_velvet_noise, noise_length)

    # Excitation signals in vocal cord vibrations and aperiodic noise were generated

    for i in range(number_of_aperiodicities):
        spec = 0.5 + 0.5 * np.cos(((w - (frequency_interval * i)) / frequency_range)  * 2 * np.pi)
        spec[w > (frequency_interval * (i + 1))] = 0
        spec[w < (frequency_interval * (i - 1))] = 0
        if i == number_of_aperiodicities - 1:
            spec[w > (frequency_interval * i)] = 1
        pulse[:,i] = fftshift(ifft(np.r_[spec, spec[-2:0:-1]]).real)
        noise[:,i] = ifft(spec_n * fft(pulse[:,i], noise_length)).real
    h = hanning(fft_size+2)[1:-1]
    pulse[:,0] = pulse[:,0] - np.mean(pulse[:,0]) * h / np.mean(h)
    return {'pulse':pulse,
            'noise':noise}

def generate_modified_velvet_noise(N, fs):
    base_period = np.array([8, 30, 60])
    short_period = 8 * round_matlab(base_period * fs / 48000)
    n = np.zeros(N + int(np.max(short_period)) + 1)

    index = 0
    while 1:
        # random.seed(10)
        v_len = random.randint(0, len(short_period)-1)
        tmp = generate_short_velvet_noise(int(short_period[v_len]))
        n[index: index + int(short_period[v_len])] = tmp
        index += int(short_period[v_len])
        if index >= N-1: break
    return n[:N]


def generate_short_velvet_noise(N):
    n = np.zeros(N)
    td = 4
    r = int(N // td + 0.5)
    safety_rand = np.ones(r)
    safety_rand[int(r//2):] *= -1
    safety_rand *= 2
    # safety_rand = 2 * np.r_[np.ones(r//2), -np.ones(r//2)]
    
    for i in range(r):
        # random.seed(10)
        tmp_index = random.randint(0, r-1)
        tmp = safety_rand[tmp_index]
        safety_rand[tmp_index] = safety_rand[i]
        safety_rand[i] = tmp
    # np.random.seed(10)
    n[td * np.arange(r) + np.random.randint(td, size=r)] = safety_rand
    return n


@numba.jit((numba.float64[:],), nopython=True, cache=True)
def round_matlab(x: np.ndarray) -> np.ndarray:
    '''
    round function works as matlab round
    :param x: input vector
    :return: rounded vector
    '''
    #return int(Decimal(n).quantize(0, ROUND_HALF_UP))
    y = x.copy()
    y[x > 0] += 0.5
    y[x <= 0] -= 0.5
    return y
