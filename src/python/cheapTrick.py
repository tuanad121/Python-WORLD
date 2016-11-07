import numpy as np
import math as m
from scipy.interpolate import interp1d
def CheapTrick(x, fs, source_object, q1=-0.09):
    f0_low_limit = 71
    default_f0 = 500
    fft_size = 2 ** m.ceil(m.log(3 * fs / f0_low_limit + 1, 2))
    temporal_positions = source_object['temporal_positions']
    f0_sequence = source_object['f0']
    f0_sequence[source_object['vuv'] == 0] = default_f0
    spectrogram = np.zeros([fft_size / 2 + 1, len(f0_sequence)])
    for i in range(len(f0_sequence)):
        spectrogram[:,i] = EstimateOneSlice(x, fs, f0_sequence[i],\
                                            temporal_positions[i], fft_size, f0_low_limit, q1)

    return 0
################################################################################################################
def EstimateOneSlice(x, fs, f0, temporal_position,\
    fft_size, f0_low_limit, q1):
    if f0 < f0_low_limit: f0 = f0_low_limit #safe guard
    waveform = CalculateWaveform(x, fs, f0, temporal_position)
    power_spectrum = CalculatePowerSpectrum(waveform, fs, fft_size, f0)
    smoothed_spectrum = LinearSmoothing(power_spectrum, f0, fs, fft_size)
    print('bang')
    #spectral_envelope = SmoothingWithRecovery(smoothed_spectrum; smoothed_spectrum(end - 1 : -1 : 2)], f0, fs,\
    #    fft_size, q1)   
    return 0
#################################################################################################################
def CalculatePowerSpectrum(waveform, fs, fft_size, f0):
    power_spectrum = np.abs(np.fft.fft(waveform, fft_size)) ** 2
    frequency_axis = np.arange(0, fft_size) / fft_size * fs
    low_frequency_axis = frequency_axis[frequency_axis < 1.2 * f0]
    low_frequency_replica = interp1d(f0 - low_frequency_axis,\
                                    power_spectrum[frequency_axis < 1.2 * f0],\
                                    fill_value='extrapolate')(low_frequency_axis)
    power_spectrum[frequency_axis < f0] =\
        low_frequency_replica[frequency_axis < f0] +\
        power_spectrum[frequency_axis < f0]
    
    power_spectrum[ -1 : fft_size / 2 : -1] = power_spectrum[1 : fft_size / 2]    
    return power_spectrum
    
##################################################################################################################
def CalculateWaveform(x, fs, f0, temporal_position):
    #  prepare internal variables
    fragment_index = np.arange(np.round(1.5 * fs / f0) + 1)
    number_of_fragments = len(fragment_index)
    base_index = np.append(-fragment_index[number_of_fragments - 1 : 0 : -1], fragment_index)
    index = temporal_position * fs + 1 + base_index
    safe_index = np.minimum(len(x), np.maximum(1, np.round(index)))
    safe_index = safe_index.astype(int)
    
    #  wave segments and set of windows preparation
    segment = x[safe_index - 1]
    time_axis = base_index / fs / 1.5 +\
        (temporal_position * fs - np.round(temporal_position * fs)) / fs    
    window = 0.5 * np.cos(m.pi * time_axis * f0) + 0.5
    window = window / np.sqrt(np.sum(window ** 2))
    waveform = segment * window - window * np.mean(segment * window) / np.mean(window)
    return waveform
def LinearSmoothing(power_spectrum, f0, fs, fft_size):
    double_frequency_axis = np.arange(0, 2 * fft_size) / fft_size * fs - fs;
    #double_spectrum = [power_spectrum; power_spectrum];
    
    #double_segment = cumsum(double_spectrum * (fs / fft_size));
    #center_frequency = (0 : fft_size / 2)' / fft_size * fs;
    #low_levels = interp1H(double_frequency_axis + fs / fft_size / 2,...
    #                      double_segment, center_frequency - f0 / 3);
    #high_levels = interp1H(double_frequency_axis + fs / fft_size / 2,...
    #                       double_segment, center_frequency + f0 / 3);
    
    #smoothed_spectrum = (high_levels - low_levels) * 1.5 / f0;    
    return 0