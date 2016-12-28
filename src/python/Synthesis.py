#3rd party imports
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
#from scipy.io.wavfile import write

#build-in imports
from decimal import Decimal, ROUND_HALF_UP
import sys
from fftfilt import fftfilt

import cython
#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.nonecheck(False)

#import numba

@cython.locals(i=cython.int)
#@numba.jit()
def Synthesis(source_object, filter_object):
    '''
    Waveform synthesis from the estimated parameters
    y = Synthesis(source_object, filter_object)
    
    Inputs
    source_object : F0 and aperiodicity
    filter_object : spectral envelope
    Output
    y : synthesized waveform
    
    '''
    vuv = source_object['vuv']
    spectrogram = filter_object['spectrogram']
    default_f0 = 500
    f0 = source_object['f0']
    fs = filter_object['fs']
    temporal_positions = source_object['temporal_positions']
    signal_time = np.arange(temporal_positions[0], temporal_positions[-1] + 1 / fs, 1 / fs)
    y = 0 * signal_time
    
    pulse_locations, pulse_locations_index, interpolated_vuv = \
        TimeBaseGeneration(temporal_positions, f0, fs, vuv, signal_time, default_f0)  
    
    fft_size = (spectrogram.shape[0] - 1) * 2
    base_index = np.arange(-fft_size // 2 + 1, fft_size // 2 + 1)
    y_length = len(y)
    tmp_complex_cepstrum = np.zeros(fft_size)
    latter_index = np.arange(fft_size // 2 + 1, fft_size + 1)
    
    temporal_position_index = interp1d(temporal_positions, \
                                       np.arange(1, len(temporal_positions) + 1), \
                                       kind='linear', fill_value='extrapolate')(pulse_locations)
    temporal_position_index = np.maximum(1, np.minimum(len(temporal_positions), \
                                                       temporal_position_index))
    
    amplitude_aperiodic = source_object['aperiodicity'] ** 2
    amplitude_periodic = np.maximum(0.001, (1 - amplitude_aperiodic))
    
    for i in range(len(pulse_locations_index)):
        
        spectrum_slice, periodic_slice, aperiodic_slice = \
            GetSpectralParameters(temporal_positions, temporal_position_index[i],\
                                  spectrogram, amplitude_periodic, amplitude_aperiodic, pulse_locations[i])
    
        noise_size = \
            pulse_locations_index[min(len(pulse_locations_index) - 1, i + 1)] - \
            pulse_locations_index[i]
        output_buffer_index = np.maximum(1, np.minimum(y_length, pulse_locations_index[i] + base_index))
        
        if interpolated_vuv[pulse_locations_index[i] - 1] >= 0.5:
            
            #tmp_periodic_spectrum = spectrum_slice * periodic_slice
            spectrum_slice *= periodic_slice

            #tmp_periodic_spectrum[tmp_periodic_spectrum == 0] = sys.float_info.epsilon
            spectrum_slice[spectrum_slice == 0] = sys.float_info.epsilon

            #periodic_spectrum = np.append(tmp_periodic_spectrum, tmp_periodic_spectrum[-2 : 0 : -1])
            periodic_spectrum = np.append(spectrum_slice, spectrum_slice[-2 : 0 : -1])


            tmp_cepstrum = np.fft.fft(np.log(np.abs(periodic_spectrum)) / 2).real
            tmp_complex_cepstrum[latter_index.astype(int) - 1] = tmp_cepstrum[latter_index.astype(int) - 1] * 2
            tmp_complex_cepstrum[0] = tmp_cepstrum[0]

            #  TODO: possible speed up via rfft?
            #tmp_cepstrum = np.fft.rfft(np.log(np.abs(periodic_spectrum)) / 2)
            #tmp_complex_cepstrum[latter_index.astype(int) - 1] = tmp_cepstrum * 2
            #tmp_complex_cepstrum[0] = tmp_cepstrum[0]


            response = np.fft.fftshift(np.fft.ifft(np.exp(np.fft.ifft(tmp_complex_cepstrum))).real)
            #y[output_buffer_index.astype(int) - 1] =\
            #    y[output_buffer_index.astype(int) - 1] + response * np.sqrt(max(1, noise_size))
            response *= np.sqrt(max(1, noise_size))
            y[output_buffer_index.astype(int) - 1] += response
            tmp_aperiodic_spectrum = spectrum_slice * aperiodic_slice
        else:
            tmp_aperiodic_spectrum = spectrum_slice
    
        tmp_aperiodic_spectrum[tmp_aperiodic_spectrum == 0] = sys.float_info.epsilon
        aperiodic_spectrum =\
            np.append(tmp_aperiodic_spectrum, tmp_aperiodic_spectrum[-2 : 0 : -1])
        tmp_cepstrum = np.real(np.fft.fft(np.log(np.abs(aperiodic_spectrum)) / 2))
        tmp_complex_cepstrum[latter_index.astype(int) - 1] = tmp_cepstrum[latter_index.astype(int) - 1] * 2
        tmp_complex_cepstrum[0] = tmp_cepstrum[0]
        response2 = np.fft.fftshift(np.real(np.fft.ifft(np.exp(np.fft.ifft(tmp_complex_cepstrum)))))
        noise_input = np.random.randn(max(3, noise_size))
        
        y[output_buffer_index - 1] += fftfilt(noise_input - np.mean(noise_input), response2)
        
    return y

#####################################################

def TimeBaseGeneration(temporal_positions, f0, fs, vuv, signal_time, default_f0):
    
    f0_interpolated_raw =\
        interp1d(temporal_positions, f0, kind='linear', fill_value='extrapolate')(signal_time)
    vuv_interpolated = \
        interp1d(temporal_positions, vuv, kind='linear', fill_value='extrapolate')(signal_time)
    vuv_interpolated = vuv_interpolated > 0.5
    f0_interpolated = f0_interpolated_raw * vuv_interpolated
    f0_interpolated[f0_interpolated == 0] = \
        f0_interpolated[f0_interpolated == 0] + default_f0

    total_phase = np.cumsum(2 * np.pi * f0_interpolated / fs)
    temp = np.diff(np.remainder(total_phase, 2 * np.pi))
    pulse_locations = signal_time[np.abs(np.r_[temp[0], temp]) > np.pi / 2] # TODO: check whether this is the correct strategy
    pulse_locations_index = np.array([int(Decimal(elm * fs).quantize(0, ROUND_HALF_UP)) for elm in pulse_locations]) + 1
    return pulse_locations, pulse_locations_index, vuv_interpolated

#####################################################

def GetSpectralParameters(temporal_positions, temporal_position_index,\
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