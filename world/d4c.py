# buil-in imports
import math

# 3rd imports
import numpy as np
from scipy.interpolate import interp1d
import numba


def d4c(x: np.ndarray, fs: int, f0_object: dict, threshold: float=0.85, fft_size_for_spectrum: int=None) -> dict:
    '''
    calculate aperiodicity
    :param x: input signal
    :param fs: sampling frequency
    :param f0_object: F0 information object
    :param threshold: used for D4C Love Train, set to 0 to use conventional D4C
    :return:
    '''
    f0_low_limit = 47
    fft_size = int(2 ** np.ceil(np.log2(4 * fs / f0_low_limit + 1)))
    f0_low_limit_for_spectrum = 71
    if fft_size_for_spectrum == None:
        fft_size_for_spectrum = int(2 ** np.ceil(np.log2(3 * fs / f0_low_limit_for_spectrum + 1)))
    upper_limit = 15000
    frequency_interval = 3000
    if fs < 16000:
        frequency_interval = 2000
    source_object = f0_object

    temporal_positions = f0_object['temporal_positions']
    f0_sequence = f0_object['f0']
    f0_sequence[f0_object['vuv'] == 0] = 0

    number_of_aperiodicity = int(np.floor(np.min([upper_limit, fs / 2 - frequency_interval]) / frequency_interval))
    assert number_of_aperiodicity > 0

    # The window function used for the CalculateFeature() is designed here to speed up
    window_length = np.floor(frequency_interval / (fs / fft_size)) * 2 + 1
    window = nuttall(window_length)

    aperiodicity = np.zeros([fft_size_for_spectrum // 2 + 1, len(f0_sequence)])
    ap_debug = np.zeros([number_of_aperiodicity, len(f0_sequence)])

    frequency_axis = np.arange(fft_size_for_spectrum / 2 + 1) * fs / fft_size_for_spectrum
    coarse_axis = np.r_[np.arange(number_of_aperiodicity + 1) * frequency_interval, fs / 2]

    for i in range(len(f0_sequence)):

        if d4c_love_train(x, fs, f0_sequence[i], temporal_positions[i], threshold) == 0:
            aperiodicity[:, i] = 1 - 0.000000000001
            continue

        current_f0 = max(f0_low_limit, f0_sequence[i])
        coarse_aperiodicity = estimate_one_slice(x, fs, current_f0, frequency_interval, temporal_positions[i], fft_size,
                                                 number_of_aperiodicity, window)
        coarse_aperiodicity = np.maximum(0, coarse_aperiodicity - (current_f0 - 100) * 2 / 100)
        ap_debug[:, i] = -coarse_aperiodicity # for debug
        aperiodicity[:, i] = 10 ** ((interp1d(coarse_axis, np.r_[np.r_[-60, -coarse_aperiodicity], -0.000000000001])
                                     (frequency_axis)) / 20)
        
    source_object['aperiodicity'] = aperiodicity
    source_object['coarse_ap'] = ap_debug
    
    return source_object


###################################################################################
def d4c_love_train(x: np.ndarray, fs: int, current_f0: float, current_position: float, threshold: float) -> int:
    vuv = 0
    if current_f0 == 0:
        return vuv

    lowest_f0 = 40
    current_f0 = max(current_f0, lowest_f0)
    fft_size = int(2 ** np.ceil(np.log2(3 * fs / lowest_f0 + 1)))
    # Cumulative powers at 100, 4000, 7900 Hz are used for VUV identification.
    boundary0 = int(np.ceil(100 / (fs / fft_size)) + 1)
    boundary1 = int(np.ceil(4000 / (fs / fft_size)) + 1)
    boundary2 = int(np.ceil(7900 / (fs / fft_size)) + 1)

    waveform = get_windowed_waveform(x, fs, current_f0, current_position, 1.5, 2)
    power_spectrum = np.abs(np.fft.fft(waveform, fft_size)) ** 2
    power_spectrum[0 : boundary0] = 0.0
    cumulative_spectrum = np.cumsum(power_spectrum)

    if (cumulative_spectrum[boundary1 - 1] / cumulative_spectrum[boundary2 - 1]) > threshold:
        vuv = 1
    return vuv


###################################################################################
def get_windowed_waveform(x: np.ndarray, fs: int, current_f0: float,
                          current_position: float, half_length: float, window_type: int) -> np.ndarray: # 1: hanning, 2: blackman
    # prepare internal variables
    half_window_length = int(half_length * fs / current_f0 + 0.5)
    base_index = np.arange(-half_window_length, half_window_length + 1)
    index = int(current_position * fs + 0.501) + 1.0 + base_index
    safe_index = np.minimum(len(x), np.maximum(1, round_matlab(index)))
    safe_index = np.array(safe_index, dtype=np.int)
    #  wave segments and set of windows preparation
    segment = x[safe_index - 1]
    time_axis = base_index / fs / half_length + \
                (current_position * fs - int(current_position * fs + 0.5)) / fs
        
    if window_type == 1: # hanning
        window = 0.5 * np.cos(np.pi * time_axis * current_f0) + 0.5
    else: # blackman
        window = 0.08 * np.cos(np.pi * time_axis * current_f0 * 2) + 0.5 * np.cos(np.pi * time_axis * current_f0) + 0.42
    waveform = segment * window - window * np.mean(segment * window) / np.mean(window)
    return waveform


###################################################################################
def estimate_one_slice(x: np.ndarray, fs: int, current_f0: float,
                       frequency_interval: int, current_position: float,
                       fft_size: int, number_of_aperiodicity: int, window: np.ndarray) -> np.ndarray:
    if current_f0 == 0:
        return np.zeros(number_of_aperiodicity)

    static_centroid =\
        get_static_centroid(x, fs, current_f0, current_position, fft_size)
    waveform = get_windowed_waveform(x, fs, current_f0, current_position, 2, 1)
    smoothed_power_spectrum = get_smoothed_power_spectrum(waveform, fs, current_f0, fft_size)
    static_group_delay = get_static_group_delay(static_centroid, smoothed_power_spectrum, fs, current_f0, fft_size)
    coarse_aperiodicity =\
        get_coarse_aperiodicity(static_group_delay, fs, fft_size, frequency_interval, number_of_aperiodicity, window)

    return coarse_aperiodicity


#########################################################################################################
def get_static_centroid(x: np.ndarray, fs: int, current_f0: float, temporal_position: float, fft_size: int) -> np.ndarray:
    '''
        First step: calculation of temporally static parameters on basis of group delay
    '''     
    waveform1 = get_windowed_waveform(x, fs, current_f0, temporal_position + 1 / current_f0 / 4, 2, 2)
    waveform2 = get_windowed_waveform(x, fs, current_f0, temporal_position - 1 / current_f0 / 4, 2, 2)

    centroid1 = get_centroid(waveform1, fft_size)
    centroid2 = get_centroid(waveform2, fft_size)

    return dc_correction(centroid1 + centroid2, fs, fft_size, current_f0)
    

#########################################################################################################
def get_centroid(x: np.ndarray, fft_size: int) -> np.ndarray:
    time_axis = np.arange(1,len(x)+1)
    x = x / np.sqrt(np.sum(x**2))

    # Centroid calculation on frequency domain.
    spectrum = np.fft.fft(x, fft_size)
    weighted_spectrum = np.fft.fft(-x * time_axis * 1j, fft_size)
    return -np.imag(weighted_spectrum) * np.real(spectrum) + np.imag(spectrum) * np.real(weighted_spectrum)


##########################################################################################################
def get_smoothed_power_spectrum(waveform: np.ndarray, fs: float, f0: float, fft_size: int) -> np.ndarray:
    power_spectrum = np.abs(np.fft.fft(waveform, fft_size)) ** 2
    spectral_envelope = dc_correction(power_spectrum, fs, fft_size, f0)
    spectral_envelope = linear_smoothing(spectral_envelope, fs, fft_size, f0)
    return np.r_[spectral_envelope, spectral_envelope[-2 : 0: -1]]


##########################################################################################################
def get_static_group_delay(static_centroid, smoothed_power_spectrum, fs, f0, fft_size):
    '''
          Second step: calculation of parameter shaping
    '''    
    group_delay = static_centroid / smoothed_power_spectrum # T_g(w)
    group_delay = linear_smoothing(group_delay, fs, fft_size, f0 / 2) # T_gs(w)
    group_delay = np.append(group_delay, group_delay[-2 : 0 : -1])
    smoothed_group_delay = linear_smoothing(group_delay, fs, fft_size, f0) # T_gb(w)
    group_delay = group_delay[0 : fft_size // 2 + 1] - smoothed_group_delay # T_D(w)
    return np.r_[group_delay, group_delay[-2 : 0 : -1]]


#########################################################################################################
def linear_smoothing(group_delay, fs, fft_size, width):
    double_frequency_axis = np.arange(2 * fft_size) / fft_size * fs - fs
    double_spectrum = np.append(group_delay, group_delay)
    
    double_segment = np.cumsum(double_spectrum * (fs / fft_size))
    center_frequency = np.arange(fft_size / 2 + 1) / fft_size * fs

    low_levels = interp1H(double_frequency_axis + fs / fft_size / 2, double_segment, center_frequency - width / 2)
    high_levels = interp1H(double_frequency_axis + fs / fft_size / 2, double_segment, center_frequency + width / 2)
    
    return (high_levels - low_levels) / width


#########################################################################################################
def get_coarse_aperiodicity(group_delay, fs, fft_size, frequency_interval, number_of_aperiodicity, window):
    '''
        Third step:
        estimation of band-aperiodicity
        '''    
    boundary = int(fft_size / len(window) * 8 + 0.5)
    
    half_window_length = int(np.floor(len(window) / 2))
    coarse_aperiodicity = np.zeros(number_of_aperiodicity)
    for i in range(int(number_of_aperiodicity)):
        center = int(np.floor(frequency_interval * (i + 1) / (fs / fft_size)))
        segment = group_delay[center - half_window_length : center + half_window_length + 1] * window
        power_spectrum = np.abs(np.fft.fft(segment, fft_size)) ** 2
    
        cumulative_power_spectrum = np.cumsum(np.sort(power_spectrum[0 : fft_size // 2 + 1]))
        coarse_aperiodicity[i] =\
            -10 * np.log10(cumulative_power_spectrum[fft_size // 2 - boundary - 1] / cumulative_power_spectrum[-1])
    return coarse_aperiodicity


#########################################################################################################
def dc_correction(signal, fs, fft_size, f0):
    frequency_axis = np.arange(fft_size) / fft_size * fs
    low_frequency_axis = frequency_axis[frequency_axis < 1.2 * f0]
    low_frequency_replica = interp1d(f0 - low_frequency_axis, signal[frequency_axis < 1.2 * f0],
                                    fill_value='extrapolate')(low_frequency_axis)
    signal[frequency_axis < f0] =\
        low_frequency_replica[frequency_axis[:len(low_frequency_replica)] < f0] + signal[frequency_axis < f0]
    
    signal[-1 : fft_size // 2 : -1] = signal[1 : fft_size // 2]
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
    '''
        Nuttall window 
        'Some windows with very good sidelobe behavior', Nuttall, 1981
        '''    
    t = np.asmatrix(np.arange(N) * 2 * math.pi / (N-1))
    coefs = np.array([0.355768, -0.487396, 0.144232, -0.012604])
    window = coefs @ np.cos(np.matrix([0,1,2,3]).T @ t)
    return np.squeeze(np.asarray(window))


#####################################################################################################
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