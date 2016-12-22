# built-in imports
import math
from decimal import Decimal, ROUND_HALF_UP

#3rd-party imports
import numpy as np
#import oct2py

def StoneMask(x, fs, temporal_positions, f0):
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
    refined_f0 = np.copy(f0);
    for i in range(len(temporal_positions)):
        if f0[i] != 0:
            refined_f0[i] = GetRefinedF0(x, fs, temporal_positions[i], f0[i])
            if abs(refined_f0[i] - f0[i]) / f0[i] > 0.2:
                refined_f0[i] = f0[i]
    return refined_f0

def GetRefinedF0(x, fs, current_time, f0_initial):
    half_window_length = np.ceil(3 * fs / f0_initial / 2)
    window_length_in_time = (2 * half_window_length + 1) / fs
    base_time = np.arange(-half_window_length, half_window_length + 1) / fs
    fft_size = 2 ** math.ceil(math.log((half_window_length * 2 + 1), 2) + 1)
    fx = (np.arange(fft_size) / fft_size * fs)
    
    #use octave round() function
    #oc = oct2py.octave
    #index_raw = oc.round((current_time + base_time) * fs) # round half up
    #index_raw = np.around((current_time + base_time) * fs) # round half to nearest even
    index_raw = np.array([int(Decimal(elm).quantize(0, ROUND_HALF_UP)) for elm in ((current_time + base_time) * fs)])
    #use numpy around() function
    
    index_time = (index_raw - 1) / fs
    window_time = index_time - current_time

    main_window = 0.42 + 0.5 * np.cos(2 * math.pi * window_time / window_length_in_time) + \
        0.08 * np.cos(4 * math.pi * window_time / window_length_in_time)
    diff_window = -(np.diff(np.append([0], main_window)) + np.diff(np.append(main_window, [0]))) / 2
    
    index = np.maximum(0, np.minimum(len(x) - 1, index_raw ))
    index = index.astype(int)
    spectrum = np.fft.fft(x[index - 1] * main_window, fft_size)
    diff_spectrum = np.fft.fft(x[index - 1] * diff_window, fft_size)
    numerator_i = np.real(spectrum) * np.imag(diff_spectrum) - np.imag(spectrum) * np.real(diff_spectrum)
    power_spectrum = np.abs(spectrum) ** 2
    instantaneous_frequency = fx + numerator_i / power_spectrum * fs / 2 / math.pi
    
    trim_index = np.array([1, 2])
    index_list_trim = np.around(f0_initial * fft_size / fs * trim_index) + 1
    index_list_trim = index_list_trim.astype(int)
    fixp_list = instantaneous_frequency[index_list_trim - 1]
    amp_list = np.sqrt(power_spectrum[index_list_trim - 1])
    f0_initial = np.sum(amp_list * fixp_list) / np.sum(amp_list * trim_index)
    
    if f0_initial < 0:
        return 0
    
    trim_index = np.array([1, 2, 3, 4, 5, 6])
    index_list_trim = np.around(f0_initial * fft_size / fs * trim_index) + 1
    index_list_trim = index_list_trim.astype(int)
    fixp_list = instantaneous_frequency[index_list_trim - 1]
    amp_list = np.sqrt(power_spectrum[index_list_trim - 1])
    refined_f0 = np.sum(amp_list * fixp_list) / np.sum(amp_list * trim_index)   
    return refined_f0