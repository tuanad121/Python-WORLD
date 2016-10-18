from pysig import track
import math
from scipy.signal import decimate
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt 

import numpy as np
SHORT_MAX = 32767 #normalize input signal
def Dio(x, fs, f0_floor = 71, f0_ceil = 800, channels_in_octave = 2, target_fs = 4000, frame_period = 5, allowed_range = 0.1):
    #print(fs)
    #print(x)
    temporal_positions = np.arange(0, np.size(x)/fs, frame_period/1000) #careful!! check later
    
    boundary_f0_list = range(math.ceil(math.log(f0_ceil/f0_floor,2)*channels_in_octave))
    boundary_f0_list = [elm + 1 for elm in boundary_f0_list]
    boundary_f0_list = [elm/channels_in_octave for elm in boundary_f0_list]
    boundary_f0_list = [2.0 ** elm for elm in boundary_f0_list]
    boundary_f0_list = np.array([f0_floor * elm for elm in boundary_f0_list])
    #down sample to target Hz
    y, actual_fs = CalculateDownsampledSignal(x, fs, target_fs) #comeback and check later, cannot check on octave
    
    y_spectrum = CalculateSpectrum(x, actual_fs, f0_floor)
    raw_f0_candidate, raw_stability = CalculateCandidateAndStabirity(np.size(temporal_positions), boundary_f0_list, np.size(y),temporal_positions, actual_fs, y_spectrum, f0_floor, f0_ceil)    
    
    f0_candidates = SortCandidates(raw_f0_candidate, raw_stability)# not test yet
    
    f0 = np.array([[1,2,3],[0,9,8]])
    f0_candidates = np.array([4,5,6,7])
    raw_f0_candidates = []
    
    vuv = []
    
    return {'f0':f0,
            'f0_candidates':f0_candidates,
            'raw_f0_candidates':raw_f0_candidates,
            'temporal_positions':temporal_positions,
            'vuv':vuv
            }

##########################################################################################################

def CalculateDownsampledSignal(x, fs, target_fs):
    decimation_ratio = int(np.round(fs / target_fs))
    if fs < target_fs:
        y = x
        y = y - np.mean(y)
        actual_fs = fs
        return y, actual_fs
    else: 
        y = decimate(x, decimation_ratio, 3)
        y = y - np.mean(y)
        actual_fs = fs/decimation_ratio
        return y, actual_fs
  
##########################################################################################################   

def CalculateSpectrum(x, fs, lowest_f0):
    fft_size = 2 ** math.ceil( math.log(np.size(x) + np.round(fs / lowest_f0 / 2) * 4,2) ) #strange
    #low-cut filtering
    cutoff_in_sample = np.round(fs / 50)
    low_cut_filter = np.hanning(2 * cutoff_in_sample + 1)
    low_cut_filter = -low_cut_filter / np.sum(low_cut_filter)
    low_cut_filter[cutoff_in_sample] = low_cut_filter[cutoff_in_sample] + 1
    low_cut_filter = np.concatenate([low_cut_filter,np.zeros(fft_size - np.size(low_cut_filter))])
    low_cut_filter = np.concatenate([low_cut_filter[cutoff_in_sample:], low_cut_filter[:cutoff_in_sample]])
    
    x_spetrum = np.fft.fft(x, fft_size) * np.fft.fft(low_cut_filter, fft_size)
    return x_spetrum

##########################################################################################################

def CalculateCandidateAndStabirity(number_of_frames, boundary_f0_list,y_length, temporal_positions, actual_fs, y_spectrum, f0_floor, f0_ceil):
    raw_f0_candidate = np.zeros(length(boundary_f0_list), number_of_frames)
    raw_f0_stability = np.zeros(length(boundary_f0_list), number_of_frames)
    for i in range(np.size(boundary_f0_list)):
        interpolated_f0, f0_deviations = CalculateRawEvent(boundary_f0_list(i), actual_fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil);
        raw_f0_stability[i, :] = math.exp(-(f0_deviations / max(0.0000001, interpolated_f0)))
        raw_f0_candidate[i, :] = interpolated_f0
    return raw_f0_candidate, raw_f0_stability

##########################################################################################################

def SortCandidates(f0_candidate_map, stability_map):
    #not test yet ***********
    number_of_candidates, number_of_frames = f0_candidate_map.shape()
    sorted_index = sort(-stability_map, axis = 0, kind='quicksort')
    f0_candidates = np.zeros(number_of_candidates, number_of_frames)
    f0_candidates_score = np.zeros(number_of_candidates, number_of_frames)    
    for i in range(number_of_frames):
        f0_candidates[:, i] = f0_candidate_map[sorted_index[1:number_of_candidates,i], i]
        f0_candidates_score[:,i] = stability_map[sorted_index[1:number_of_candidates,i], i]    
    return f0_candidates, f0_candidates_score 

########################################################################################################## 

def CalculateRawEvent(boundary_f0,fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil):
    half_filter_length = np.round(fs / boundary_f0 / 2);
    low_pass_filter = nuttall(half_filter_length * 4);
    index_bias = low_pass_filter.argmax(axis=0)
    
    spectrum_low_pass_filter = np.fft.fft(low_pass_filter, length(y_spectrum))
    
    filtered_signal = np.real(np.fft.ifft(spectrum_low_pass_filter * y_spectrum))
    filtered_signal = filtered_signal(index_bias + np.arange(y_length)) 
    
    # calculate 4 kinds of event
    negative_zero_cross = ZeroCrossingEngine(filtered_signal, fs)
    positive_zero_cross = ZeroCrossingEngine(-filtered_signal, fs)
    peak = ZeroCrossingEngine(diff(filtered_signal), fs)
    dip = ZeroCrossingEngine(-diff(filtered_signal), fs)
    
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
        interpolated_f0_list[0,:] = interp1d(negative_zero_cross['interval_locations'], negative_zero_cross['interval_based_f0'], fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[1,:] = interp1d(positive_zero_cross['interval_locations'], positive_zero_cross['interval_based_f0'], fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[2,:] = interp1d(peak['interval_locations'], peak['interval_based_f0'], fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[3,:] = interp1d(dip['interval_locations'], dip['interval_based_f0'], fill_value='extrapolate')(temporal_positions)
        
        interpolated_f0 = np.mean(interpolated_f0_list, axis=0)
        f0_deviations = np.std(interpolated_f0_list, axis=0)
    else:
        interpolated_f0 = temporal_positions * 0;
        f0_deviations = temporal_positions * 0 + 1000;        
    return interpolated_f0, f0_deviations

#negative zero crossing: going from positive to negative
##########################################################################################################
def ZeroCrossingEngine(x, fs):
    #y=x.tolist()
    negative_going_points = []
    
    for i in range(np.size(x)-1):
        if x[i]*x[i+1] < 0 and x[i] > x[i+1]:
            negative_going_points.append(i)
    edge_list = np.array(negative_going_points)# the different indexing between MATLAB and Python make this variable different, maybe OK!?
    fine_edge_list = edge_list - x[edge_list] / (x[edge_list + 1] - x[edge_list]);
    interval_locations =(fine_edge_list[0 : np.size(fine_edge_list)-1] + fine_edge_list[1:]) / 2 / fs
    interval_based_f0 = fs / np.diff(fine_edge_list)
    
    return {'interval_locations':interval_locations, 
            'interval_based_f0':interval_based_f0
            }

##########################################################################################################

def nuttall(N):
    t = np.asmatrix(np.arange(N) * 2 * math.pi / (N-1))
    coefs = np.array([0.355768, -0.487396, 0.144232, -0.012604])
    window = np.dot(coefs, np.cos(np.dot(np.matrix([0,1,2,3]).T,t)))
    return np.asarray(window)

name='test/test-mwm'
inp_wav = track.Wave.wavread('%s.wav' % name, channel=0)

#print(inp_wav.get_value())
#results=Dio(inp_wav.get_value()/SHORT_MAX, inp_wav.fs)
#frame_period=5
#temporal_positions = np.arange(0, np.size(inp_wav.get_value())/inp_wav.fs, frame_period/1000)

#negative_zero_cross = ZeroCrossingEngine(inp_wav.get_value()/SHORT_MAX, inp_wav.fs);
#positive_zero_cross = ZeroCrossingEngine(-inp_wav.get_value()/SHORT_MAX, inp_wav.fs);
#peak = ZeroCrossingEngine(np.diff(inp_wav.get_value()/SHORT_MAX), inp_wav.fs);
#dip = ZeroCrossingEngine(-np.diff(inp_wav.get_value()/SHORT_MAX), inp_wav.fs);
#f0_candidates, f0_deviations = GetF0Candidates(negative_zero_cross, positive_zero_cross, peak, dip, 
#               temporal_positions)


print('done')
#print(results['temporal_positions'])



