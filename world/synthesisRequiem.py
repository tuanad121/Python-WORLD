# 3rd-party imports
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hanning
from scipy.fftpack import fft, ifft
import numba
import matplotlib.pyplot as plt

# build-in imports
from decimal import Decimal, ROUND_HALF_UP

def synthesisRequiem(source_object, filter_object, seeds_signals):
    excitation_signal = get_excitation_signal(source_object['temporal_positions'],
                                              filter_object['fs'],
                                              source_object['f0'],
                                              source_object['vuv'],
                                              seeds_signals['pulse'],
                                              seeds_signals['noise'],
                                              source_object['aperiodicity'])
    y = get_waveform(excitation_signal,
                     filter_object['spectrogram'],
                     source_object['temporal_positions'],
                     source_object['f0'],
                     filter_object['fs'])
    return y

def get_excitation_signal(temporal_positions,
                          fs,
                          f0,
                          vuv,
                          pulse_seed,
                          noise_seed,
                          band_aperiodicity):

    fft_size = pulse_seed.shape[0]
    base_index = np.arange(-fft_size // 2 + 1, fft_size // 2 + 1)
    number_of_aperiodicities = pulse_seed.shape[1]

    time_axis = np.arange(temporal_positions[0], temporal_positions[-1] + 1 / fs, 1 / fs)
    periodic_component = np.zeros(len(time_axis))
    aperiodic_component = np.zeros(len(time_axis))

    pulse_locations_index, interpolated_vuv = time_base_generation(temporal_positions, f0, fs, vuv, time_axis)

    # band-aperiodicity is resampled at sampling frequency of fs Hz
    interpolated_aperiodicity = aperiodicity_generation(temporal_positions, band_aperiodicity, time_axis)

    # generation of the aperiodic component
    for i in range(number_of_aperiodicities):
        noise = generate_noise(len(aperiodic_component), noise_seed, i)
        aperiodic_component += (noise * interpolated_aperiodicity[i, :len(aperiodic_component)])

    # generation of the periodic component
    for i in range(len(pulse_locations_index)):
        if (interpolated_vuv[pulse_locations_index[i]-1] <= 0.5) or (interpolated_aperiodicity[0, pulse_locations_index[i]-1] > 0.999):
            continue
        noise_size = pulse_locations_index[min(len(pulse_locations_index) - 1, i + 1)] - pulse_locations_index[i]
        noise_size = np.sqrt(max(1, noise_size))
        output_buffer_index = np.maximum(1, np.minimum(len(time_axis), pulse_locations_index[i] + base_index))
        response = get_one_periodic_excitation(number_of_aperiodicities, pulse_seed, interpolated_aperiodicity[:, pulse_locations_index[i]-1], noise_size)
        periodic_component[output_buffer_index.astype(int)-1] += response
    excitation_signal = periodic_component + aperiodic_component
    return excitation_signal


def get_one_periodic_excitation(number_of_aperiodicities, pulse_seed, aperiodicity, noise_size):
    response = np.zeros(len(pulse_seed[:,0]))
    for i in range(number_of_aperiodicities):
        response += pulse_seed[:,i] * (1 - aperiodicity[i])
    response *= noise_size
    return response


def get_waveform(excitation_signal, spectrogram, temporal_positions, f0, fs):
    y = np.zeros(len(excitation_signal))
    fft_size = (spectrogram.shape[0] - 1) * 2
    latter_index = np.arange(int(fft_size // 2 + 1), fft_size+1)
    frame_period_sample = int((temporal_positions[1] - temporal_positions[0]) * fs)
    win_len = frame_period_sample * 2 - 1
    half_win_len = frame_period_sample - 1
    win = hanning(win_len+2)[1:-1]

    for i in range(2, len(f0)-1):
        origin = (i - 1) * frame_period_sample - half_win_len
        safe_index = np.minimum(len(y), np.arange(origin, origin + win_len))

        tmp = excitation_signal[safe_index-1] * win
        spec = spectrogram[:,i-1]
        periodic_spectrum = np.r_[spec, spec[-2:0:-1]]

        tmp_cepstrum = np.fft.fft(np.log(np.abs(periodic_spectrum)) / 2).real
        tmp_complex_cepstrum = np.zeros(fft_size)
        tmp_complex_cepstrum[latter_index.astype(int) - 1] = tmp_cepstrum[latter_index.astype(int) - 1] * 2
        tmp_complex_cepstrum[0] = tmp_cepstrum[0]

        spectrum = np.exp(np.fft.ifft(tmp_complex_cepstrum))
        response = ifft(spectrum * fft(tmp, fft_size)).real

        safe_index = np.minimum(len(y), np.arange(origin, origin+fft_size))
        y[safe_index-1] += response
    return y


def time_base_generation(temporal_positions, f0, fs, vuv, time_axis):
    f0_interpolated_raw = interp1d(temporal_positions, f0, kind='linear', fill_value='extrapolate')(time_axis)
    vuv_interpolated = interp1d(temporal_positions, vuv, kind='linear', fill_value='extrapolate')(time_axis)
    vuv_interpolated = vuv_interpolated > 0.5

    f0_interpolated = f0_interpolated_raw * vuv_interpolated
    default_f0 = 500
    f0_interpolated[f0_interpolated == 0] = f0_interpolated[f0_interpolated == 0] + default_f0

    total_phase = np.cumsum(2 * np.pi * f0_interpolated / fs)
    wrap_phase = np.remainder(total_phase, 2 * np.pi)
    pulse_locations = (time_axis[:-1])[np.abs(np.diff(wrap_phase)) > np.pi]
    pulse_locations_index = np.array([int(Decimal(elm * fs).quantize(0, ROUND_HALF_UP)) for elm in pulse_locations]) + 1

    return pulse_locations_index, vuv_interpolated

def aperiodicity_generation(temporal_positions, band_aperiodicity, time_axis):
    number_of_aperiodicities = band_aperiodicity.shape[0]
    multi_aperiodicity = np.zeros((number_of_aperiodicities, len(time_axis)))

    for i in range(number_of_aperiodicities):
        multi_aperiodicity[i,:] = interp1d(temporal_positions, 10 ** (band_aperiodicity[i, :] / 10),
                                           kind='linear', fill_value='extrapolate')(time_axis)

    return multi_aperiodicity


def generate_noise(N, noise_seed, frequency_band):
    # current_index is a persistent variable of the function
    if np.all(generate_noise.current_index == None):
        generate_noise.current_index = np.zeros(noise_seed.shape[1])
    noise_length = noise_seed.shape[0]

    index = np.remainder(np.arange(generate_noise.current_index[frequency_band], generate_noise.current_index[frequency_band]+N), noise_length).astype(int)
    n = noise_seed[index, frequency_band]
    generate_noise.current_index[frequency_band] = index[-1]
    return n
generate_noise.current_index = None

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
