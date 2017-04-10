#build-in imports
import math as m
from decimal import Decimal, ROUND_HALF_UP

#3rd party imports
import numpy as np
from scipy.interpolate import interp1d


def cheaptrick(x, fs, source_object, q1=-0.15):
    '''
    Generate smooth spectrogram from signal x, eliminating the affect of fundamental frequency F0
    Input:
    x: ndarray of signal (samples)
    fs: sampling rate
    source_object: generated using dio.py
    Output:
    A dictionary contains information spectrograms and framming.
    '''
    f0_low_limit = 71
    default_f0 = 500
    fft_size = 2 ** np.ceil(np.log2(3 * fs / f0_low_limit + 1))

    f0_low_limit = fs * 3.0 / (fft_size - 3.0)
    temporal_positions = source_object['temporal_positions']
    f0_sequence = source_object['f0']
    f0_sequence[source_object['vuv'] == 0] = default_f0
    
    spectrogram = np.zeros([int(fft_size // 2) + 1, len(f0_sequence)])
    pitch_syn_spectrogram = 1j * np.zeros([int(fft_size), len(f0_sequence)])
    for i in range(len(f0_sequence)):
        if f0_sequence[i] < f0_low_limit:
            f0_sequence[i] = default_f0
        spectrogram[:,i],  pitch_syn_spectrogram[:, i]= estimate_one_slice(x, fs, f0_sequence[i], temporal_positions[i], fft_size, q1)
    return {'temporal_positions': temporal_positions,
            'spectrogram': spectrogram,
            'fs': fs,
            'ps spectrogram': pitch_syn_spectrogram
            }


################################################################################################################
def estimate_one_slice(x, fs, current_f0, current_position, fft_size, q1):
    '''
    Calculate spectrum using CheapTrick algorithm consisting 3 steps
    '''
    # First step: F0-adaptive windowing
    waveform = calculate_windowed_waveform(x, fs, current_f0, current_position)
    power_spectrum, pitch_syn_spectrum = get_power_spectrum(waveform, fs, fft_size, current_f0)
    # Second step: Frequency domain smoothing
    smoothed_spectrum = linear_smoothing(power_spectrum, current_f0, fs, fft_size)
    # Third step: Liftering in quefrency domain
    spectral_envelope = smoothing_with_recovery(np.append(smoothed_spectrum, smoothed_spectrum[-2 : 0 : -1]), current_f0, fs,
                                                fft_size, q1)
    return spectral_envelope, pitch_syn_spectrum


#################################################################################################################
def get_power_spectrum(waveform, fs, fft_size, f0):
    pitch_syn_spectrum = np.fft.fft(waveform, fft_size)
    power_spectrum = np.abs(pitch_syn_spectrum) ** 2
    frequency_axis = np.arange(fft_size) / fft_size * fs
    low_frequency_axis = frequency_axis[frequency_axis < f0 + fs / fft_size]
    low_frequency_replica = interp1d(f0 - low_frequency_axis, power_spectrum[frequency_axis < f0 + fs / fft_size],
                                     fill_value='extrapolate')(low_frequency_axis)
    power_spectrum[frequency_axis < f0] =\
        low_frequency_replica[frequency_axis[:len(low_frequency_replica)] < f0] + power_spectrum[frequency_axis < f0]
    
    power_spectrum[-1:int(fft_size // 2): -1] = power_spectrum[1: int(fft_size // 2)]
    return power_spectrum, pitch_syn_spectrum


##################################################################################################################
def calculate_windowed_waveform(x, fs, f0, temporal_position):
    '''
    First step: F0-adaptive windowing
    Design a window function with basic idea of pitch-synchronous analysis.
    A Hanning window with length 3*T0 is used.
    Using the window makes over all power of the periodic signal temporally stable 
    '''
    half_window_length = round_matlab(1.5 * fs / f0)
    base_index = np.arange(-half_window_length, half_window_length + 1)
    index = round_matlab(temporal_position * fs + 0.001) + 1.0 + base_index
    safe_index = np.minimum(len(x), np.maximum(1, np.array([round_matlab(elm) for elm in index])))
    
    #  wave segments and set of windows preparation
    segment = x[safe_index - 1]
    time_axis = base_index / fs / 1.5
    window = 0.5 * np.cos(np.pi * time_axis * f0) + 0.5
    window /= np.sqrt(np.sum(window ** 2))
    waveform = segment * window - window * np.mean(segment * window) / np.mean(window)
    return waveform


###################################################################################################################
def linear_smoothing(power_spectrum, f0, fs, fft_size):
    '''
        Second step: Frequency domain smoothing of power spectrum
        This step is carried out to ensure that the power spectrum has no zeros.
        It avoids log(0) in next step
    '''    
    double_frequency_axis = np.arange(2 * fft_size) / fft_size * fs - fs
    double_spectrum = np.r_[power_spectrum, power_spectrum]

    double_segment = np.cumsum(double_spectrum * (fs / fft_size))
    center_frequency = np.arange(fft_size // 2 + 1) / fft_size * fs
    low_levels = interp1H(double_frequency_axis + fs / fft_size / 2, double_segment, center_frequency - f0 / 3)
    high_levels = interp1H(double_frequency_axis + fs / fft_size / 2, double_segment, center_frequency + f0 / 3)
    smoothed_spectrum = (high_levels - low_levels) * 1.5 / f0
    return smoothed_spectrum


####################################################################################################################
def interp1H(x, y, xi):
    delta_x = x[1] - x[0]
    xi = np.maximum(x[0], np.minimum(x[-1], xi))
    xi_base = np.floor((xi - x[0]) / delta_x)
    xi_fraction = (xi - x[0]) / delta_x - xi_base
    delta_y = np.append(np.diff(y), 0)
    yi = y[xi_base.astype(int)] + delta_y[xi_base.astype(int)] * xi_fraction
    return yi


####################################################################################################################
def smoothing_with_recovery(smoothed_spectrum, f0, fs, fft_size, q1):
    '''
        Third step: Liftering in quefrency domain
        Remove frequency fluctuation caused by dicretization
        '''    
    quefrency_axis = np.arange(fft_size) / fs
    # smoothing liftering function: l_s(T)
    smoothing_lifter = np.r_[1, np.sin(np.pi * f0 * quefrency_axis[1:]) / (np.pi * f0 * quefrency_axis[1:])]
    smoothing_lifter[int(fft_size // 2) + 1:] =\
        smoothing_lifter[round_matlab(fft_size / 2) - 1 : 0 : -1]
    # liftering function for spectral recovery: l_q(T)
    compensation_lifter =\
        (1 - 2 * q1) + 2 * q1 * np.cos(2 * np.pi * quefrency_axis * f0)
    compensation_lifter[int(fft_size // 2) + 1 : ] =\
        compensation_lifter[int(fft_size // 2) - 1: 0 : -1]
    tandem_cepstrum = np.fft.fft(np.log(smoothed_spectrum))
    tmp_spectral_envelope =\
        np.exp(np.real(np.fft.ifft(tandem_cepstrum * smoothing_lifter * compensation_lifter)))
    spectral_envelope = tmp_spectral_envelope[: int(fft_size // 2) + 1]
    return spectral_envelope


#####################################################################################################################
def round_matlab(n):
    '''
    this function works as Matlab round() function
    python round function choose the nearest even number to n, which is different from Matlab round function
    :param n: input number
    :return: rounded n
    '''
    return int(Decimal(n).quantize(0, ROUND_HALF_UP))