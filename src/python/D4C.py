# buil-in imports
import math
from decimal import Decimal, ROUND_HALF_UP
import copy

# 3rd imports
import numpy as np
from scipy.interpolate import interp1d

def D4C(x, fs, f0_object):
    f0_low_limit = 71
    fft_size = 2 ** np.ceil(np.log2(4 * fs / f0_low_limit + 1))
    fft_size_for_spectrum = 2 ** np.ceil(np.log2(3 * fs / f0_low_limit + 1))
    upper_limit = 15000
    frequency_interval = 3000
    source_object = f0_object

    temporal_positions = f0_object['temporal_positions']
    f0_sequence = f0_object['f0']
    f0_sequence[f0_object['vuv'] == 0] = 0

    number_of_aperiodicity = \
    np.floor(np.min([upper_limit, fs / 2 - frequency_interval]) / frequency_interval)

    # The window function used for the CalculateFeature() is designed here to
    # speed up
    window_length = np.floor(frequency_interval / (fs / fft_size)) * 2 + 1
    window = nuttall(window_length)

    aperiodicity = np.zeros([fft_size_for_spectrum / 2 + 1, len(f0_sequence)])
    ap_debug = np.zeros([number_of_aperiodicity, len(f0_sequence)])

    frequency_axis = np.arange(fft_size_for_spectrum / 2 + 1) * fs / fft_size_for_spectrum
    coarse_axis = np.arange(number_of_aperiodicity + 1) * frequency_interval
    coarse_axis = np.append(coarse_axis, fs/2)

    for i in range(len(f0_sequence)):
        if f0_sequence[i] == 0:
            aperiodicity[:, i] = 0
            continue
        coarse_aperiodicity = EstimateOneSlice(x, fs, f0_sequence[i], \
                                           frequency_interval, temporal_positions[i], fft_size, \
                                           number_of_aperiodicity, window)
        coarse_aperiodicity = np.maximum(0, coarse_aperiodicity - (f0_sequence[i] - 100) * 2 / 100)
        ap_debug[:, i] = coarse_aperiodicity # for debug
        aperiodicity[:, i] = 10 ** ((interp1d(coarse_axis, \
                                              np.append(np.append(-60, -coarse_aperiodicity), 0), \
                                              kind='linear')(frequency_axis)) / 20)
        
    source_object['aperiodicity'] = aperiodicity
    source_object['coarse_ap'] = ap_debug
    
    return source_object

###################################################################################

def CalculateWaveform(x, fs, f0, temporal_position,\
    half_length, window_type): # 1: hanning, 2: blackman
    # prepare internal variables
    fragment_index = np.arange(int(Decimal(half_length * fs / f0).quantize(0, ROUND_HALF_UP)) + 1)
    number_of_fragments = len(fragment_index)
    base_index = np.append(-fragment_index[number_of_fragments - 1 : 0 : -1], fragment_index)
    index = temporal_position * fs + 1 + base_index
    safe_index = np.minimum(len(x), np.maximum(1, [int(Decimal(elm).quantize(0, ROUND_HALF_UP)) for elm in index]))  

    #  wave segments and set of windows preparation
    segment = x[safe_index - 1]
    time_axis = base_index / fs / half_length + \
                (temporal_position * fs - \
                 int(Decimal(temporal_position * fs).quantize(0, ROUND_HALF_UP))) / fs 
        
    if window_type == 1: # hanning
        window = 0.5 * np.cos(np.pi * time_axis * f0) + 0.5
    else: # blackman
        window = 0.08 * np.cos(np.pi * time_axis * f0 * 2) +\
                 0.5 * np.cos(np.pi * time_axis * f0) + 0.42
    waveform = segment * window - window * np.mean(segment * window) / np.mean(window)
    return waveform


###################################################################################
def EstimateOneSlice(x, fs, f0, \
                 frequency_interval, temporal_position, \
                 fft_size, number_of_aperiodicity, window):
    if f0 == 0:
        return np.zeros(number_of_aperiodicity)

    static_centroid =\
        CalculateStaticCentroid(x, fs, f0, temporal_position, fft_size)
    waveform = CalculateWaveform(x, fs, f0, temporal_position, 2, 1)
    smoothed_power_spectrum =\
        CalculateSmoothedPowerSpectrum(waveform, fs, f0, fft_size)
    static_group_delay =\
        CalculateStaticGroupDelay(static_centroid, smoothed_power_spectrum, fs, f0,\
                                  fft_size)
    coarse_aperiodicity =\
        CalculateCoarseAperiodicity(static_group_delay, fs, fft_size,\
                                    frequency_interval, number_of_aperiodicity, window)
    return coarse_aperiodicity

#########################################################################################################

def CalculateStaticCentroid(x, fs, f0, temporal_position,\
                            fft_size):
    waveform1 =\
        CalculateWaveform(x, fs, f0, temporal_position + 1 / f0 / 4, 2, 2)
    waveform2 =\
        CalculateWaveform(x, fs, f0, temporal_position - 1 / f0 / 4, 2, 2)
    centroid1 = CalculateCentroid(waveform1, fft_size)
    centroid2 = CalculateCentroid(waveform2, fft_size)
    centroid = DCCorrection(centroid1 + centroid2, fs, fft_size, f0)
    return centroid
    
#########################################################################################################
    
def CalculateCentroid(x, fft_size):
    time_axis = np.arange(1,len(x)+1)
    x = x / np.sqrt(np.sum(x**2))

    # Centroid calculation on frequency domain.
    spectrum = np.fft.fft(x, fft_size)
    weighted_spectrum = np.fft.fft(-x * time_axis * 1j, fft_size)
    centroid = -np.imag(weighted_spectrum) * np.real(spectrum) + \
    np.imag(spectrum) * np.real(weighted_spectrum)
    return centroid

##########################################################################################################

def CalculateSmoothedPowerSpectrum(waveform, fs,\
                                   f0, fft_size):
    power_spectrum = np.abs(np.fft.fft(waveform, fft_size)) ** 2
    spectral_envelope = DCCorrection(power_spectrum, fs, fft_size, f0)
    spectral_envelope = LinearSmoothing(spectral_envelope, fs, fft_size, f0)
    spectral_envelope = np.append(spectral_envelope, spectral_envelope[-2 : 0: -1])
    return spectral_envelope

##########################################################################################################

def CalculateStaticGroupDelay(static_centroid,\
                              smoothed_power_spectrum, fs, f0, fft_size):
    group_delay = static_centroid / smoothed_power_spectrum
    group_delay = LinearSmoothing(group_delay, fs, fft_size, f0 / 2)
    group_delay = np.append(group_delay, group_delay[-2 : 0 : -1])
    smoothed_group_delay = LinearSmoothing(group_delay, fs, fft_size, f0)
    group_delay = group_delay[0 : fft_size / 2 + 1] - smoothed_group_delay
    group_delay = np.append(group_delay, group_delay[-2 : 0 : -1])
    return group_delay

#########################################################################################################

def LinearSmoothing(group_delay, fs, fft_size, width):
    double_frequency_axis = np.arange(2 * fft_size) / fft_size * fs - fs
    double_spectrum = np.append(group_delay, group_delay)
    
    double_segment = np.cumsum(double_spectrum * (fs / fft_size))
    center_frequency = np.arange(fft_size / 2 + 1) / fft_size * fs
    low_levels = interp1H(double_frequency_axis + fs / fft_size / 2,\
                          double_segment, center_frequency - width / 2)
    high_levels = interp1H(double_frequency_axis + fs / fft_size / 2,\
                           double_segment, center_frequency + width / 2)
    
    smoothed_group_delay = (high_levels - low_levels) / width
    return smoothed_group_delay

#########################################################################################################

def CalculateCoarseAperiodicity(group_delay, fs,\
                                fft_size, frequency_interval, number_of_aperiodicity, window):
    boundary = int(Decimal(fft_size / len(window) * 8).quantize(0, ROUND_HALF_UP))
    
    
    half_window_length = np.floor(len(window) / 2)
    coarse_aperiodicity = np.zeros(number_of_aperiodicity)
    for i in range(int(number_of_aperiodicity)):
        center = np.floor(frequency_interval * (i + 1) / (fs / fft_size))
        segment = group_delay[center - half_window_length :\
                               center + half_window_length + 1] * window
        power_spectrum = np.abs(np.fft.fft(segment, fft_size)) ** 2
    
        cumulative_power_spectrum =\
            np.cumsum(np.sort(power_spectrum[0 : fft_size / 2 + 1]))
        coarse_aperiodicity[i] =\
            -10 * np.log10(cumulative_power_spectrum[fft_size / 2 - boundary - 1] / \
                        cumulative_power_spectrum[-1])
    return coarse_aperiodicity

#########################################################################################################

def DCCorrection(signal, fs, fft_size, f0):
    frequency_axis = np.arange(fft_size) / fft_size * fs
    low_frequency_axis = frequency_axis[frequency_axis < 1.2 * f0]
    low_frequency_replica = interp1d(f0 - low_frequency_axis,\
                                     signal[frequency_axis < 1.2 * f0],\
                                    fill_value='extrapolate')(low_frequency_axis)
    signal[frequency_axis < f0] =\
        low_frequency_replica[frequency_axis < f0] + signal[frequency_axis < f0]
    
    signal[-1 : fft_size / 2 + 2 - 2 : -1] = signal[1 : fft_size / 2]
    return signal

##########################################################################################################

def interp1H(x, y, xi):
    delta_x = x[1] - x[0]
    xi = np.maximum(x[0], np.minimum(x[-1], xi))
    xi_base = np.floor((xi - x[0]) / delta_x)
    xi_fraction = (xi - x[0]) / delta_x - xi_base
    delta_y = np.append(np.diff(y), 0)
    yi = y[xi_base.astype(int)] + delta_y[xi_base.astype(int)] * xi_fraction
    return yi
    
##########################################################################################################

def nuttall(N):
    t = np.asmatrix(np.arange(N) * 2 * math.pi / (N-1))
    coefs = np.array([0.355768, -0.487396, 0.144232, -0.012604])
    window = coefs @ np.cos(np.matrix([0,1,2,3]).T @ t)
    return np.squeeze(np.asarray(window))