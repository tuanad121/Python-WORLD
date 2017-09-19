# built-in imports
import math

#3rd-party imports
import numpy as np
import numba

def stonemask(x, fs, temporal_positions, f0):
    '''
    Refine F0 by instantaneous frequency
    refined_f0 = StoneMask(x, fs, temporal_positions, f0)
    
    Inputs
      x  : input signal
      fs : sampling frequency
      temporal_positions : Temporal positions in each f0
      f0 : F0 estimated from an estimator.
     Output
       refined_f0 : Refined f0    
    '''
    refined_f0 = np.copy(f0)
    for i in range(len(temporal_positions)):
        if f0[i] != 0:
            refined_f0[i] = get_refined_f0(x, fs, temporal_positions[i], f0[i])
            if abs(refined_f0[i] - f0[i]) / f0[i] > 0.2:
                refined_f0[i] = f0[i]
    return refined_f0


def get_refined_f0(x, fs, current_time, current_f0):
    f0_initial = current_f0
    half_window_length = np.ceil(3 * fs / f0_initial / 2)
    window_length_in_time = (2 * half_window_length + 1) / fs
    base_time = np.arange(-half_window_length, half_window_length + 1) / fs
    fft_size = 2 ** math.ceil(math.log((half_window_length * 2 + 1), 2) + 1)
    fx = (np.arange(fft_size) / fft_size * fs)

    base_time = np.array([float("{0:.4f}".format(elm)) for elm in base_time])
    index_raw = round_matlab((current_time + base_time) * fs)
    
    index_time = (index_raw - 1) / fs
    window_time = index_time - current_time

    main_window = 0.42 + 0.5 * np.cos(2 * math.pi * window_time / window_length_in_time) + \
        0.08 * np.cos(4 * math.pi * window_time / window_length_in_time)
    diff_window = -(np.diff(np.r_[0, main_window]) + np.diff(np.r_[main_window, 0])) / 2
    
    index = np.maximum(1, np.minimum(len(x), index_raw ))
    index = np.array(index, dtype=np.int)
    spectrum = np.fft.fft(x[index - 1] * main_window, fft_size)
    diff_spectrum = np.fft.fft(x[index - 1] * diff_window, fft_size)
    numerator_i = np.real(spectrum) * np.imag(diff_spectrum) - np.imag(spectrum) * np.real(diff_spectrum)
    power_spectrum = np.abs(spectrum) ** 2
    from sys import float_info
    power_spectrum[power_spectrum == 0] = float_info.epsilon
    instantaneous_frequency = fx + numerator_i / power_spectrum * fs / 2 / math.pi
    
    trim_index = np.array([1, 2])
    index_list_trim = round_matlab(f0_initial * fft_size / fs * trim_index) + 1
    index_list_trim = np.array(index_list_trim, np.int)
    fixp_list = instantaneous_frequency[index_list_trim - 1]
    amp_list = np.sqrt(power_spectrum[index_list_trim - 1])
    f0_initial = np.sum(amp_list * fixp_list) / np.sum(amp_list * trim_index)

    if f0_initial < 0:
        return 0
    
    trim_index = np.array([1, 2, 3, 4, 5, 6])
    
    index_list_trim = round_matlab(f0_initial * fft_size / fs * trim_index) + 1
    index_list_trim = np.array(index_list_trim, np.int)
    fixp_list = instantaneous_frequency[index_list_trim - 1]
    amp_list = np.sqrt(power_spectrum[index_list_trim - 1])
    refined_f0 = np.sum(amp_list * fixp_list) / np.sum(amp_list * trim_index)

    return refined_f0

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