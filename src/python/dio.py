# built-in imports
import math
from decimal import Decimal, ROUND_HALF_UP

# 3rd-party imports
from scipy.signal import resample, decimate
from scipy.interpolate import interp1d

import numpy as np
def Dio(x, fs, f0_floor=71, f0_ceil=800, channels_in_octave=2, target_fs=4000, frame_period=5, allowed_range=0.1):
    '''
    F0 estimation by DIO
    f0_parameter = Dio(x, fs, f0_ceil, channels_in_octave, target_fs, frame_period, allowed_range);
    
    Inputs
    x  : input signal
    fs : sampling frequency
    other settings : f0_floor (Hz), f0_ceil (Hz), target_fs (Hz)
             channels_in_octave (ch), allowed_range, and frame_period (ms)
    Output
    f0 infromation

    Caution: minimum frame_period is 1.
    '''
    temporal_positions = np.arange(0, np.size(x) / fs, frame_period / 1000) #careful!! check later
    
    boundary_f0_list = np.arange(math.ceil(np.log2(f0_ceil / f0_floor) * channels_in_octave)) + 1
    
    boundary_f0_list = boundary_f0_list / channels_in_octave 
    boundary_f0_list = 2.0 ** boundary_f0_list
    boundary_f0_list = f0_floor * boundary_f0_list
    
    #down sample to target Hz
    y, actual_fs = CalculateDownsampledSignal(x, fs, target_fs)    

    y_spectrum = CalculateSpectrum(y, actual_fs, f0_floor)
    raw_f0_candidate, raw_stability = CalculateCandidateAndStabirity(np.size(temporal_positions), \
                                                                     boundary_f0_list, np.size(y), \
                                                                     temporal_positions, actual_fs, \
                                                                     y_spectrum, f0_floor, f0_ceil)    
    
    f0_candidates, _ = SortCandidates(raw_f0_candidate, raw_stability)
    f0_candidates_tmp = np.copy(f0_candidates)#just want to keep original values of f0_candidates, maybe we don't need this line
    f0, vuv = FixF0Contour(f0_candidates, frame_period, f0_floor, allowed_range)
    
    return {'f0':f0,
            'f0_candidates':f0_candidates_tmp,
            'raw_f0_candidates':raw_f0_candidate,
            'temporal_positions':temporal_positions,
            'vuv':vuv
            }

##########################################################################################################

def CalculateDownsampledSignal(x, fs, target_fs):
    decimation_ratio = int(Decimal(fs / target_fs).quantize(0, ROUND_HALF_UP))
    if fs < target_fs:
        y = x
        y = y - np.mean(y)
        actual_fs = fs
        return y, actual_fs
    else: 
        y = decimate(x, decimation_ratio, ftype='iir', n = 3, zero_phase=True)
        y = y - np.mean(y)
        actual_fs = fs/decimation_ratio
        
        return y, actual_fs
  
##########################################################################################################   

def CalculateSpectrum(x, fs, lowest_f0):
    fft_size = 2 ** math.ceil(math.log(np.size(x) + \
                                        int(Decimal(fs / lowest_f0 / 2).quantize(0, ROUND_HALF_UP)) * 4,2)) 
    #low-cut filtering
    cutoff_in_sample = int(Decimal(fs / 50).quantize(0, ROUND_HALF_UP))
    low_cut_filter = np.hanning(2 * cutoff_in_sample + 1)
    low_cut_filter = -low_cut_filter / np.sum(low_cut_filter)
    low_cut_filter[cutoff_in_sample] = low_cut_filter[cutoff_in_sample] + 1
    low_cut_filter = np.concatenate([low_cut_filter,np.zeros(fft_size - np.size(low_cut_filter))])
    low_cut_filter = np.concatenate([low_cut_filter[cutoff_in_sample:], low_cut_filter[:cutoff_in_sample]])
    
    x_spetrum = np.fft.fft(x, fft_size) * np.fft.fft(low_cut_filter, fft_size)
    return x_spetrum

##########################################################################################################

def CalculateCandidateAndStabirity(number_of_frames, \
                                   boundary_f0_list,y_length, \
                                   temporal_positions, \
                                   actual_fs, y_spectrum, \
                                   f0_floor, \
                                   f0_ceil):
    raw_f0_candidate = np.zeros((np.size(boundary_f0_list), number_of_frames))
    raw_f0_stability = np.zeros((np.size(boundary_f0_list), number_of_frames))
    for i in range(np.size(boundary_f0_list)):
        interpolated_f0, f0_deviations = CalculateRawEvent(boundary_f0_list[i], \
                                                           actual_fs, y_spectrum, \
                                                           y_length, temporal_positions, \
                                                           f0_floor, \
                                                           f0_ceil)
        
        raw_f0_stability[i, :] = np.exp(-(f0_deviations / np.maximum(interpolated_f0, 0.0000001)))
        raw_f0_candidate[i, :] = interpolated_f0
    return raw_f0_candidate, raw_f0_stability

##########################################################################################################

def SortCandidates(f0_candidate_map, stability_map):
    #not test yet ***********
    number_of_candidates, number_of_frames = f0_candidate_map.shape
    sorted_index = np.argsort(-stability_map, axis=0, kind='quicksort')
    f0_candidates = np.zeros((number_of_candidates, number_of_frames))
    f0_candidates_score = np.zeros((number_of_candidates, number_of_frames))    
    for i in range(number_of_frames):
        f0_candidates[:, i] = f0_candidate_map[sorted_index[:number_of_candidates,i], i]
        f0_candidates_score[:,i] = stability_map[sorted_index[:number_of_candidates,i], i]
    #print('type of f0_candidates {}'.format(type(f0_candidates)))
    #print('type of f0_candidates_score {}'.format(type(f0_candidates_score)))
    return f0_candidates, f0_candidates_score 

########################################################################################################## 

def CalculateRawEvent(boundary_f0, fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil):
    half_filter_length = int(Decimal(fs / boundary_f0 / 2).quantize(0, ROUND_HALF_UP))
    low_pass_filter = nuttall(half_filter_length * 4)
    index_bias = low_pass_filter.argmax()
    spectrum_low_pass_filter = np.fft.fft(low_pass_filter, len(y_spectrum))
    
    filtered_signal = np.real(np.fft.ifft(spectrum_low_pass_filter * y_spectrum))
    
    filtered_signal = filtered_signal[index_bias + np.arange(y_length)] 
    
    # calculate 4 kinds of event
    negative_zero_cross = ZeroCrossingEngine(filtered_signal, fs)
    positive_zero_cross = ZeroCrossingEngine(-filtered_signal, fs)
    peak = ZeroCrossingEngine(np.diff(filtered_signal), fs)
    dip = ZeroCrossingEngine(-np.diff(filtered_signal), fs)
    
    f0_candidate, f0_deviations = GetF0Candidates(negative_zero_cross, positive_zero_cross, peak, dip, temporal_positions)
    # remove untrustful candidates
    f0_candidate[f0_candidate > boundary_f0] = 0;
    f0_candidate[f0_candidate < (boundary_f0 / 2)] = 0;
    f0_candidate[f0_candidate > f0_ceil] = 0;
    f0_candidate[f0_candidate < f0_floor] = 0;
    f0_deviations[f0_candidate == 0] = 100000; #rough safe guard    
    
    return f0_candidate, f0_deviations

##########################################################################################################

def GetF0Candidates(negative_zero_cross, positive_zero_cross, peak, dip, temporal_positions):
    #test this one 
    usable_channel = max(0, np.size(negative_zero_cross['interval_locations']) - 2) *\
        max(0, np.size(positive_zero_cross['interval_locations']) - 2) *\
        max(0, np.size(peak['interval_locations']) - 2) *\
        max(0, np.size(dip['interval_locations']) - 2) 
    
    interpolated_f0_list = np.zeros((4, np.size(temporal_positions)))
    
    if usable_channel > 0:
        interpolated_f0_list[0,:] = interp1d(negative_zero_cross['interval_locations'], \
                                             negative_zero_cross['interval_based_f0'], \
                                             fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[1,:] = interp1d(positive_zero_cross['interval_locations'], \
                                             positive_zero_cross['interval_based_f0'], \
                                             fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[2,:] = interp1d(peak['interval_locations'], \
                                             peak['interval_based_f0'], \
                                             fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[3,:] = interp1d(dip['interval_locations'], \
                                             dip['interval_based_f0'], \
                                             fill_value='extrapolate')(temporal_positions)
        interpolated_f0 = np.mean(interpolated_f0_list, axis=0)
        f0_deviations = np.std(interpolated_f0_list, axis=0, ddof=1)
    else:
        interpolated_f0 = temporal_positions * 0;
        f0_deviations = temporal_positions * 0 + 1000;        
    return interpolated_f0, f0_deviations

##########################################################################################################
#negative zero crossing: going from positive to negative
def ZeroCrossingEngine(x, fs):
    #y=x.tolist()
    negative_going_points = []
    
    for i in range(np.size(x) - 1):
        if x[i]*x[i+1] < 0 and x[i] > x[i+1]:
            negative_going_points.append(i)
    edge_list = np.array(negative_going_points)
    
    fine_edge_list = (edge_list) - x[edge_list] / (x[edge_list + 1] - x[edge_list]);
    
    interval_locations = (fine_edge_list[:np.size(fine_edge_list) - 1] + fine_edge_list[1:]) / 2 / fs
    interval_based_f0 = fs / np.diff(fine_edge_list)
    return {
        'interval_locations':interval_locations, 
        'interval_based_f0':interval_based_f0
            }

##########################################################################################################

def nuttall(N):
    t = np.asmatrix(np.arange(N) * 2 * math.pi / (N-1))
    coefs = np.array([0.355768, -0.487396, 0.144232, -0.012604])
    window = coefs @ np.cos(np.matrix([0,1,2,3]).T @ t)
    return np.squeeze(np.asarray(window))

##########################################################################################################

def FixF0Contour(f0_candidates, frame_period, f0_floor, \
                                  allowed_range):
# if abs((f0(n) - f0(n+1)) / f0(n)) exceeds this value,
# f0(n) is not reliable.
# F0 is continuous at least voice_range_minimum (sample)
    voice_range_minimum =int(Decimal(1 / (frame_period / 1000) / f0_floor).quantize(0, ROUND_HALF_UP)) * 2 + 1
    f0_step1 = FixStep1(f0_candidates, voice_range_minimum, allowed_range)
    f0_step2 = FixStep2(f0_step1, voice_range_minimum)
    section_list = CountNumberOfVoicedSections(f0_step2)
    
    f0_step3 = FixStep3(f0_step2, f0_candidates, section_list, allowed_range)
    
    f0_step4 = FixStep4(f0_step3, f0_candidates, section_list, allowed_range)
    
    f0 = np.copy(f0_step4)
    vuv = np.copy(f0)
    vuv[vuv != 0] = 1
    return f0, vuv

##########################################################################################################
#Step 1: Rapid change of F0 is replaced by zeros
def FixStep1(f0_candidates, voice_range_minimum, allowed_range):
    f0_base = f0_candidates[0]
    f0_base[ : voice_range_minimum] = 0
    f0_base[-voice_range_minimum : ] = 0
    
    f0_step1 = np.copy(f0_base)
    
    for i in np.arange(voice_range_minimum - 1, len(f0_base)):
        if abs((f0_base[i] - f0_base[i-1]) / (0.000001 + f0_base[i])) > allowed_range:
            f0_step1[i] = 0
    return f0_step1

##########################################################################################################
#Step2: short-time voiced period (under voice_range_minimum) is replaced by 0
def FixStep2(f0_step1, voice_range_minimum):
    f0_step2 = np.copy(f0_step1)
    for i in np.arange((voice_range_minimum - 1) / 2 , len(f0_step1) - (voice_range_minimum - 1) / 2).astype(int):
        for j in np.arange( -(voice_range_minimum - 1) / 2 , (voice_range_minimum - 1) / 2 + 1).astype(int):
            if f0_step1[i + j] == 0:
                f0_step2[i] = 0
                break
    return f0_step2

##########################################################################################################
#Step3: short-time voiced period (under voice_range_minimum) is replaced by 0
def FixStep3(f0_step2, f0_candidates, section_list,\
                             allowed_range):  
    f0_step3 = np.copy(f0_step2)
    for i in np.arange(section_list.shape[0]):
        if i == section_list.shape[0] - 1:
            limit = len(f0_step3) - 1
        else:
            limit = section_list[i + 1, 0] + 1
        #print(len(np.arange(section_list[i, 1], limit)))
        for j in np.arange(section_list[i, 1], limit).astype(int):
            f0_step3[j + 1] = SelectBestF0(f0_step3[j], f0_step3[j - 1],\
                                           f0_candidates[:, j + 1], allowed_range)
            if f0_step3[j + 1] == 0:
                break
    return f0_step3

##########################################################################################################
# Step3: short-time voiced period (under voice_range_minimum) is replaced by 0
def FixStep4(f0_step3, f0_candidates, section_list, \
                             allowed_range):
    f0_step4 = np.copy(f0_step3)
    
    for i in range(section_list.shape[0] - 1, -1 , -1):
        if i == 0:
            limit = 1
        else:
            limit = section_list[i - 1, 1]
        for j in np.arange(section_list[i, 0], limit - 1,  -1).astype(int):
            f0_step4[j - 1] = SelectBestF0(f0_step4[j], f0_step4[j + 1], f0_candidates[:, j - 1], allowed_range)
            if f0_step4[j - 1] == 0:
                break
    return f0_step4

##########################################################################################################
def SelectBestF0(current_f0, past_f0, candidates, allowed_range):
    reference_f0 = (current_f0 * 3 - past_f0) / 2
    minimum_error = abs(reference_f0 - candidates[0])
    best_f0 = candidates[0]
    
    for i in range(1, len(candidates)):
        current_error = abs(reference_f0 - candidates[i])
        if current_error < minimum_error:
            minimum_error = current_error
            best_f0 = candidates[i]
    if abs(1 - best_f0 / reference_f0) > allowed_range:
        best_f0 = 0
    return best_f0


##########################################################################################################

def CountNumberOfVoicedSections(f0):
    vuv = np.copy(f0)
    vuv[vuv != 0] = 1
    diff_vuv = np.diff(vuv)
    boundary_list = np.append(np.append([0], np.where(diff_vuv != 0)[0]), [len(vuv) - 2])# take care of len(vuv) - 2 or len(vuv) - 1
    
    first_section = np.ceil(-0.5 * diff_vuv[boundary_list[1]])
    number_of_voiced_sections = np.floor((len(boundary_list) - (1 - first_section)) / 2).astype(int)
    voiced_section_list = np.zeros((number_of_voiced_sections, 2))
    for i in range(number_of_voiced_sections):
        voiced_section_list[i, :] = np.array([1 + boundary_list[int((i - 1) * 2 + 1 + (1 - first_section)) + 1], 
                                              boundary_list[int((i * 2) + (1 - first_section)) + 1]])
    return voiced_section_list