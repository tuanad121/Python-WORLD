#3rd party imports
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
#from scipy.io.wavfile import write

#build-in imports
from decimal import Decimal, ROUND_HALF_UP
import sys
#from .fftfilt import fftfilt

import cython
#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.nonecheck(False)

#import numba

@cython.locals(i=cython.int)
#@numba.jit()
def synthesis(source_object, filter_object):
    '''
    Waveform synthesis from the estimated parameters
    y = Synthesis(source_object, filter_object)
    
    Inputs
    source_object : F0 and aperiodicity
    filter_object : spectral envelope
    Output
    y : synthesized waveform
    
    '''
    default_f0 = 500
    vuv = source_object['vuv']
    f0 = source_object['f0']
    fs = filter_object['fs']
    spectrogram = filter_object['spectrogram']
    temporal_positions = source_object['temporal_positions']
    time_axis = np.arange(temporal_positions[0], temporal_positions[-1] + 1 / fs, 1 / fs)
    y = np.zeros(len(time_axis))
    pulse_locations, pulse_locations_index, pulse_locations_time_shift, interpolated_vuv = \
        time_base_generation(temporal_positions, f0, fs, vuv, time_axis, default_f0)
    
    fft_size = (spectrogram.shape[0] - 1) * 2
    base_index = np.arange(-fft_size // 2 + 1, fft_size // 2 + 1)
    y_length = len(y)
    tmp_complex_cepstrum = np.zeros(fft_size)
    latter_index = np.arange(fft_size // 2 + 1, fft_size + 1)
    
    temporal_position_index = interp1d(temporal_positions, np.arange(1, len(temporal_positions) + 1),
                                       fill_value='extrapolate')(pulse_locations)
    temporal_position_index = np.maximum(1, np.minimum(len(temporal_positions), temporal_position_index))
    
    amplitude_aperiodic = source_object['aperiodicity'] ** 2
    amplitude_periodic = np.maximum(0.001, (1 - amplitude_aperiodic))

    dc_remover_base = signal.hanning(fft_size + 2)[1:-1]
    dc_remover_base = dc_remover_base / np.sum(dc_remover_base)
    coefficient = 2.0 * np.pi * fs / fft_size
    
    for i in range(len(pulse_locations_index)):
        spectrum_slice, periodic_slice, aperiodic_slice = \
            get_spectral_parameters(temporal_positions, temporal_position_index[i],
                                    spectrogram, amplitude_periodic, amplitude_aperiodic, pulse_locations[i])

        noise_size = pulse_locations_index[min(len(pulse_locations_index) - 1, i + 1)] - pulse_locations_index[i]
        output_buffer_index = np.maximum(1, np.minimum(y_length, pulse_locations_index[i] + base_index))
        
        if interpolated_vuv[pulse_locations_index[i] - 1] >= 0.5 and aperiodic_slice[0] <= 0.999:
            response = get_periodic_response(spectrum_slice, periodic_slice,
                                             fft_size, latter_index, pulse_locations_time_shift[i], coefficient)
            dc_remover = dc_remover_base * -np.sum(response)
            periodic_response = response + dc_remover
            y[output_buffer_index.astype(int) - 1] += periodic_response * np.sqrt(max(1, noise_size))
            tmp_aperiodic_spectrum = spectrum_slice * aperiodic_slice
        else:
            tmp_aperiodic_spectrum = spectrum_slice
    
        tmp_aperiodic_spectrum[tmp_aperiodic_spectrum == 0] = sys.float_info.epsilon
        aperiodic_response = get_aperiodic_response(tmp_aperiodic_spectrum, fft_size, latter_index, noise_size)
        y[output_buffer_index.astype(int) - 1] += aperiodic_response
    return y

#####################################################

def get_aperiodic_response(tmp_aperiodic_spectrum, fft_size, latter_index, noise_size):
    aperiodic_spectrum = np.r_[tmp_aperiodic_spectrum, tmp_aperiodic_spectrum[-2: 0: -1]]
    tmp_cepstrum = np.fft.fft((np.log(np.abs(aperiodic_spectrum)) / 2)).real
    tmp_complex_cepstrum = np.zeros(fft_size)
    tmp_complex_cepstrum[latter_index.astype(int) - 1] = tmp_cepstrum[latter_index.astype(int) - 1] * 2
    tmp_complex_cepstrum[0] = tmp_cepstrum[0]
    response = np.fft.fftshift(np.fft.ifft(np.exp(np.fft.ifft(tmp_complex_cepstrum))).real)
    noise_input = np.random.randn(max(3, noise_size))
    # noise_input = np.zeros(max(3, noise_size)) + 0.1
    response = fftfilt(noise_input - np.mean(noise_input), response)
    return response

#######################################################

def get_periodic_response(spectrum_slice, periodic_slice, fft_size, latter_index, fractionsl_time_shift, coefficient):
    tmp_spectrum_slice = spectrum_slice * periodic_slice
    tmp_spectrum_slice[tmp_spectrum_slice == 0] = sys.float_info.epsilon

    periodic_spectrum = np.r_[tmp_spectrum_slice, tmp_spectrum_slice[-2: 0: -1]]

    tmp_cepstrum = np.fft.fft(np.log(np.abs(periodic_spectrum)) / 2).real
    tmp_complex_cepstrum = np.zeros(fft_size)
    tmp_complex_cepstrum[latter_index.astype(int) - 1] = tmp_cepstrum[latter_index.astype(int) - 1] * 2
    tmp_complex_cepstrum[0] = tmp_cepstrum[0]

    spectrum = np.exp(np.fft.ifft(tmp_complex_cepstrum))
    spectrum = spectrum[0: int(fft_size / 2) + 1]
    spectrum *= np.exp(-1j * coefficient * fractionsl_time_shift * np.arange(int(fft_size / 2) + 1))
    spectrum = np.r_[spectrum, spectrum[-2: 0: -1].conj()]
    response = np.fft.fftshift(np.fft.ifft(spectrum).real)
    return response

#######################################################

def time_base_generation(temporal_positions, f0, fs, vuv, signal_time, default_f0):
    f0_interpolated_raw = interp1d(temporal_positions, f0, kind='linear', fill_value='extrapolate')(signal_time)
    vuv_interpolated = interp1d(temporal_positions, vuv, kind='linear', fill_value='extrapolate')(signal_time)
    vuv_interpolated = vuv_interpolated > 0.5

    f0_interpolated = f0_interpolated_raw * vuv_interpolated
    f0_interpolated[f0_interpolated == 0] = f0_interpolated[f0_interpolated == 0] + default_f0

    total_phase = np.cumsum(2 * np.pi * f0_interpolated / fs)
    wrap_phase = np.remainder(total_phase, 2 * np.pi)
    pulse_locations = (signal_time[:-1])[np.abs(np.diff(wrap_phase)) > np.pi]
    assert (len(pulse_locations)) > 0
    pulse_locations_index = np.array([int(Decimal(elm * fs).quantize(0, ROUND_HALF_UP)) for elm in pulse_locations]) + 1

    y1 = wrap_phase[pulse_locations_index - 1] -2.0 * np.pi
    y2 = wrap_phase[pulse_locations_index]
    x = -y1 / (y2 - y1)

    pulse_locations_time_shift = x / fs

    return pulse_locations, pulse_locations_index, pulse_locations_time_shift, vuv_interpolated


#####################################################
def get_spectral_parameters(temporal_positions, temporal_position_index,
                            spectrogram, amplitude_periodic, amplitude_random, pulse_locations):
    floor_index = int(np.floor(temporal_position_index)) - 1
    ceil_index  = int(np.ceil(temporal_position_index)) - 1
    t1 = temporal_positions[floor_index]
    t2 = temporal_positions[ceil_index]

    x = max(t1, min(t2, pulse_locations))

    if t1 == t2:
        spectrum_slice = spectrogram[:, floor_index]
        periodic_slice = amplitude_periodic[:, floor_index]
        aperiodic_slice = amplitude_random[:, floor_index]
    else:
        #t = np.append(t1, t2)
        b = (x - t1) / (t2 - t1)
        assert 0 <= b <= 1
        a = 1 - b
        spectrum_slice = a * spectrogram[:, floor_index] + \
                         b * spectrogram[:, ceil_index]
            #interp1d(t, np.hstack([spectrogram[:, floor_index].reshape(-1, 1), \
            #                       spectrogram[:, ceil_index].reshape(-1, 1)])) \
            #(max(t1, min(t2, pulse_locations)))
        
        periodic_slice = a * amplitude_periodic[:, floor_index] + \
                         b * amplitude_periodic[:, ceil_index]
            #interp1d(t, np.hstack([amplitude_periodic[:, floor_index].reshape(-1, 1), \
            #                       amplitude_periodic[:, ceil_index].reshape(-1, 1)])) \
            #(max(t1, min(t2, pulse_locations)))
        
        aperiodic_slice = a * amplitude_random[:, floor_index] + \
                          b * amplitude_random[:, ceil_index]
            #interp1d(t, np.hstack([amplitude_random[:, floor_index].reshape(-1, 1), \
            #                       amplitude_random[:, ceil_index].reshape(-1, 1)])) \
            #(max(t1, min(t2, pulse_locations)))
    
    return spectrum_slice, periodic_slice, aperiodic_slice

###############################
def nextpow2(x):
    """Return the first integer N such that 2**N >= abs(x)"""

    return np.ceil(np.log2(abs(x)))

######################################
def fftfilt(b, x, *n):
    """Filter the signal x with the FIR filter described by the
    coefficients in b using the overlap-add method. If the FFT
    length n is not specified, it and the overlap-add block length
    are selected so as to minimize the computational cost of
    the filtering operation."""

    N_x = len(x)
    N_b = len(b)

    # Determine the FFT length to use:
    if len(n):

        # Use the specified FFT length (rounded up to the nearest
        # power of 2), provided that it is no less than the filter
        # length:
        n = n[0]
        if n != int(n) or n <= 0:
            raise ValueError('n must be a nonnegative integer')
        if n < N_b:
            n = N_b
        N_fft = 2**nextpow2(n)
    else:

        if N_x > N_b:

            # When the filter length is smaller than the signal,
            # choose the FFT length and block size that minimize the
            # FLOPS cost. Since the cost for a length-N FFT is
            # (N/2)*log2(N) and the filtering operation of each block
            # involves 2 FFT operations and N multiplications, the
            # cost of the overlap-add method for 1 length-N block is
            # N*(1+log2(N)). For the sake of efficiency, only FFT
            # lengths that are powers of 2 are considered:
            N = 2**np.arange(np.ceil(np.log2(N_b)),np.floor(np.log2(N_x)))
            cost = np.ceil(N_x/(N-N_b+1))*N*(np.log2(N)+1)
            assert len(cost) > 0
            N_fft = N[np.argmin(cost)]

        else:

            # When the filter length is at least as long as the signal,
            # filter the signal using a single block:
            N_fft = 2**nextpow2(N_b+N_x-1)

    N_fft = int(N_fft)

    # Compute the block length:
    L = int(N_fft - N_b + 1)

    # Compute the transform of the filter:
    H = np.fft.fft(b,N_fft)

    y = np.zeros(N_x,float)
    i = 0
    while i <= N_x:
        il = min([i+L,N_x])
        k = min([i+N_fft,N_x])
        yt = np.fft.ifft(np.fft.fft(x[i:il],N_fft)*H,N_fft).real # Overlap..
        y[i:k] = y[i:k] + yt[:k-i]            # and add
        i += L
    return y