# 3rd-party imports
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import lfilter

#build-in imports
from decimal import Decimal, ROUND_HALF_UP
import copy
import math

# local imports
from decimate import decimate # the script comes from scipy.signal with a little change


def havest(x, fs, f0_floor=71, f0_ceil=800, frame_period=5):
    basic_frame_period = 1
    target_fs = 8000
    basic_temporal_positions = np.arange(0, len(x) / fs, basic_frame_period / 1000 )
    channels_in_octave = 40
    f0_floor_adjusted = f0_floor * 0.9
    f0_ceil_adjusted = f0_ceil * 1.1

    boundary_f0_list = np.arange(math.ceil(np.log2(f0_ceil_adjusted / f0_floor_adjusted) * channels_in_octave)) + 1
    boundary_f0_list = boundary_f0_list / channels_in_octave
    boundary_f0_list = 2.0 ** boundary_f0_list
    boundary_f0_list *= f0_floor_adjusted

    # down - sampling to target_fs Hz
    [y, actual_fs] = CalculateDownsampledSignal(x, fs, target_fs)
    fft_size = int(2 ** np.ceil(np.log2(len(y) + round_matlab(fs / f0_floor_adjusted * 4) + 1)))
    y_spectrum = np.fft.fft(y, fft_size)

    raw_f0_candidates = CalculateCandidates(len(basic_temporal_positions), boundary_f0_list, len(y),
                                            basic_temporal_positions, actual_fs, y_spectrum, f0_floor, f0_ceil)
    print('finish calculate candidates')
    f0_candidates, number_of_candidates = DetectCandidates(raw_f0_candidates)
    print('finish dectect candidates')
    f0_candidates = OverlapF0Candidates(f0_candidates, number_of_candidates)
    print('finish overlap candidates')
    f0_candidates, f0_candidates_score = RefineCandidates(y, actual_fs,
                                                          basic_temporal_positions, f0_candidates, f0_floor, f0_ceil)
    print('finish refine candidates')
    f0_candidates, f0_candidates_score = RemoveUnreliableCandidates(f0_candidates, f0_candidates_score)

    connected_f0, vuv = FixF0Contour(f0_candidates, f0_candidates_score)
    smoothed_f0 = SmoothF0(connected_f0)
    temporal_positions = np.arange(0, len(x) / fs, frame_period / 1000)
    print('haha')
    return {
        'temporal_positions': temporal_positions,
        'f0': smoothed_f0[np.minimum(len(smoothed_f0) - 1, np.array([round_matlab(elm) for elm in temporal_positions * 1000]))],
        'vuv': vuv[np.minimum(len(smoothed_f0) - 1, np.array([round_matlab(elm) for elm in temporal_positions * 1000]))]
    }


############################################################################################
def CalculateDownsampledSignal(x, fs, target_fs):
    decimation_ratio = round_matlab(fs / target_fs)
    offset = np.ceil(140 / decimation_ratio) * decimation_ratio
    x = np.append(np.append(np.ones(offset) * x[0], x), np.ones(offset) * x[-1])

    if fs < target_fs:
        y = copy.deepcopy(x)
        actual_fs = fs
    else:
        # decimate can be troublesome
        y0 = decimate(x, decimation_ratio, ftype='iir', n = 3, zero_phase=True)
        actual_fs = fs / decimation_ratio
        y = y0[offset / decimation_ratio : -offset / decimation_ratio]
    y -= np.mean(y)
    return y, actual_fs


###################################################################################################
def CalculateCandidates(number_of_frames,
                        boundary_f0_list, y_length, temporal_positions, actual_fs, y_spectrum,
                        f0_floor, f0_ceil):
    raw_f0_candidates = np.zeros((len(boundary_f0_list), number_of_frames))

    for i in range(len(boundary_f0_list)):
        raw_f0_candidates[i, :] = \
                CalculateRawEvent(boundary_f0_list[i], actual_fs, y_spectrum,
                              y_length, temporal_positions, f0_floor, f0_ceil)
    return raw_f0_candidates


####################################################################################################
def DetectCandidates(raw_f0_candidates):
    number_of_channels, number_of_frames = raw_f0_candidates.shape
    f0_candidates = np.zeros((round_matlab(number_of_channels / 10), number_of_frames))
    number_of_candidates = 0
    threshold = 10

    for i in np.arange(number_of_frames):
        tmp = copy.deepcopy(raw_f0_candidates[:, i])
        tmp[tmp > 0] = 1
        tmp[0] = 0
        tmp[-1] = 0
        tmp = np.diff(tmp)
        st = np.where(tmp == 1)[0]
        ed = np.where(tmp == -1)[0]
        count = 0
        for j in np.arange(len(st)):
            dif = ed[j] - st[j]
            if dif >= threshold:
                tmp_f0 = raw_f0_candidates[st[j] + 1: ed[j] + 1, i]
                f0_candidates[count, i] = np.mean(tmp_f0)
                count += 1
        number_of_candidates = max(number_of_candidates, count)
    return f0_candidates, number_of_candidates


####################################################################################################
def OverlapF0Candidates(f0_candidates, max_candidates):
    n = 3 # This is the optimzied parameter.

    number_of_candidates = n * 2 + 1
    new_f0_candidates = np.zeros((number_of_candidates * max_candidates, f0_candidates.shape[1]))
    new_f0_candidates[0, :] = f0_candidates[number_of_candidates - 1, :]
    for i in np.arange(number_of_candidates):
        st1 = max(-(i - n) + 1, 1)
        ed1 = min(-(i - n), 0)
        new_f0_candidates[np.arange(max_candidates) + i * max_candidates, st1 - 1 : new_f0_candidates.shape[1] + ed1] = \
            f0_candidates[np.arange(max_candidates), -ed1 : new_f0_candidates.shape[1] - (st1 - 1)]
    return new_f0_candidates


####################################################################################################
def RefineCandidates(x, fs, temporal_positions, f0_candidates, f0_floor, f0_ceil):
    new_f0_candidates = copy.deepcopy(f0_candidates)
    f0_candidates_score = f0_candidates * 0

    for i in np.arange(len(temporal_positions)):
        for j in np.arange(f0_candidates.shape[0]):
            tmp_f0 = f0_candidates[j, i]
            if tmp_f0 == 0:
                continue
            new_f0_candidates[j, i], f0_candidates_score[j, i] =\
                GetRefinedF0(x, fs, temporal_positions[i], tmp_f0, f0_floor, f0_ceil)
    return new_f0_candidates, f0_candidates_score


####################################################################################################
def GetRefinedF0(x, fs, current_time, current_f0, f0_floor, f0_ceil):
    half_window_length = np.ceil(3 * fs / current_f0 / 2)
    window_length_in_time = (2 * half_window_length + 1) / fs
    base_time = np.arange(-half_window_length, half_window_length + 1) / fs
    fft_size = int(2 ** np.ceil(np.log2((half_window_length * 2 + 1)) + 1))
    fx = (np.arange(fft_size) / fft_size * fs)

    # First-aid treatment
    index_raw = np.array([round_matlab(elm) for elm in ((current_time + base_time) * fs + 0.001)])
    index_time = (index_raw - 1) / fs
    window_time = index_time - current_time
    main_window = 0.42 + 0.5 * np.cos(2 * math.pi * window_time / window_length_in_time) +\
                  0.08 * np.cos(4 * math.pi * window_time / window_length_in_time)
    diff_window = -(np.diff(np.append(0, main_window)) + np.diff(np.append(main_window, 0))) / 2

    index = np.maximum(1, np.minimum(len(x), index_raw)) - 1
    spectrum = np.fft.fft(x[index] * main_window, fft_size)
    diff_spectrum = np.fft.fft(x[index] * diff_window, fft_size)
    numerator_i = np.real(spectrum) * np.imag(diff_spectrum) - np.imag(spectrum) * np.real(diff_spectrum)
    power_spectrum = np.abs(spectrum) ** 2
    instantaneous_frequency = fx + numerator_i / power_spectrum * fs / 2 / math.pi

    max_trim = min(np.floor(fs / 2 / current_f0), 6) # with safe guard
    trim_index = np.arange(1, max_trim + 1)
    index_list_trim = np.array([round_matlab(elm) for elm in (current_f0 * fft_size / fs * trim_index)]) # check later
    fixp_list = instantaneous_frequency[index_list_trim]
    amp_list = np.sqrt(power_spectrum[index_list_trim])
    refined_f0 = np.sum(amp_list * fixp_list) / np.sum(amp_list * trim_index)

    variation = np.abs(((fixp_list / trim_index) - current_f0) / current_f0)
    refined_score = 1 / (0.000000000001 + np.mean(variation))
    if refined_f0 < f0_floor or refined_f0 > f0_ceil or refined_score < 2.5:
        refined_f0 = 0
        refined_score = 0
    return refined_f0, refined_score


####################################################################################################
def RemoveUnreliableCandidates(f0_candidates, f0_candidates_score):
    new_f0_candidates = copy.deepcopy(f0_candidates)
    new_f0_candidates_score = copy.deepcopy(f0_candidates_score)
    threshold = 0.05

    f0_length = f0_candidates.shape[1]
    number_of_candidates = f0_candidates.shape[0]

    for i in np.arange(1, f0_length - 1):
        for j in np.arange(0, number_of_candidates):
            reference_f0 = f0_candidates[j, i]
            if reference_f0 == 0:
                continue
            _, min_error1 = SelectBestF0(reference_f0, f0_candidates[:, i + 1], 1)
            _, min_error2 = SelectBestF0(reference_f0, f0_candidates[:, i - 1], 1)
            min_error = min([min_error1, min_error2])
            if min_error > threshold:
                new_f0_candidates[j, i] = 0
                new_f0_candidates_score[j, i] = 0
    return new_f0_candidates, new_f0_candidates_score


###################################################################################################
def SelectBestF0(reference_f0, f0_candidates, allowed_range):
    best_f0 = 0
    best_error = copy.deepcopy(allowed_range)

    for i in np.arange(len(f0_candidates)):
        tmp = np.abs(reference_f0 - f0_candidates[i]) / reference_f0
        if tmp > best_error:
            continue
        best_f0 = f0_candidates[i]
        best_error = tmp
    return best_f0, best_error


####################################################################################################
def CalculateRawEvent(boundary_f0, fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil):
    filter_length_half = int(Decimal(fs / boundary_f0 * 2).quantize(0, ROUND_HALF_UP))
    band_pass_filter_base = nuttall(filter_length_half * 2 + 1)
    shifter = np.cos(2 * math.pi * boundary_f0 * np.arange(-filter_length_half, filter_length_half + 1) / fs)
    band_pass_filter = band_pass_filter_base * shifter

    index_bias = filter_length_half + 1
    spectrum_low_pass_filter = np.fft.fft(band_pass_filter, len(y_spectrum))

    filtered_signal = np.real(np.fft.ifft(spectrum_low_pass_filter * y_spectrum))
    filtered_signal = filtered_signal[index_bias + np.arange(y_length)]

    # calculate 4 kinds of event
    negative_zero_cross = ZeroCrossingEngine(filtered_signal, fs)
    positive_zero_cross = ZeroCrossingEngine(-filtered_signal, fs)
    peak = ZeroCrossingEngine(np.diff(filtered_signal), fs)
    dip = ZeroCrossingEngine(-np.diff(filtered_signal), fs)

    f0_candidates = GetF0Candidates(negative_zero_cross, positive_zero_cross,
                                    peak, dip, temporal_positions)

    f0_candidates[f0_candidates > boundary_f0 * 1.1] = 0
    f0_candidates[f0_candidates < boundary_f0 * 0.9] = 0
    f0_candidates[f0_candidates > f0_ceil] = 0
    f0_candidates[f0_candidates < f0_floor] = 0

    return f0_candidates


###################################################################################################
# negative zero crossing: going from positive to negative
def ZeroCrossingEngine(x, fs):
    negative_going_points = np.arange(1, len(x) + 1) * \
                            ((np.append(x[1:], x[-1]) * x < 0) * (np.append(x[1:], x[-1]) < x))

    edge_list = negative_going_points[negative_going_points > 0]

    fine_edge_list = (edge_list) - x[edge_list - 1] / (x[edge_list] - x[edge_list - 1])

    interval_locations = (fine_edge_list[:np.size(fine_edge_list) - 1] + fine_edge_list[1:]) / 2 / fs
    interval_based_f0 = fs / np.diff(fine_edge_list)
    return {
        'interval_locations': interval_locations,
        'interval_based_f0': interval_based_f0
    }


####################################################################################################
def FixF0Contour(f0_candidates, f0_candidates_score):
    f0_base = SearchF0Base(f0_candidates, f0_candidates_score)
    f0_step1 = FixStep1(f0_base, 0.008) # optimized
    f0_step2 = FixStep2(f0_step1, 6) # optimized
    f0_step3 = FixStep3(f0_step2, f0_candidates, 0.18, f0_candidates_score) # optimized
    f0 = FixStep4(f0_step3, 9) # optimized
    vuv = copy.deepcopy(f0)
    vuv[vuv != 0] = 1
    return f0, vuv


####################################################################################################
# F0s with the highest score are selected as a basic f0 contour.
def SearchF0Base(f0_candidates, f0_candidates_score):
    f0_base = np.zeros((f0_candidates.shape[1]))
    for i in range(len(f0_base)):
        max_index = np.argmax(f0_candidates_score[:, i])
        f0_base[i] = f0_candidates[max_index, i]
    return f0_base


####################################################################################################
# Step 1: Rapid change of f0 contour is replaced by 0
def FixStep1(f0_base, allowed_range):
    f0_step1 = copy.copy(f0_base)
    f0_step1[0] = 0
    f0_step1[1] = 0

    for i in np.arange(2, len(f0_base)):
        if f0_base[i] == 0:
            continue
        reference_f0 = f0_base[i - 1] * 2 - f0_base[i - 2]
        if np.abs((f0_base[i] - reference_f0) / reference_f0) > allowed_range and \
                        np.abs((f0_base[i] - f0_base[i - 1]) / f0_base[i - 1]) > allowed_range:
            f0_step1[i] = 0
    return f0_step1


####################################################################################################
# Step 2: Voiced sections with a short period are removed
def FixStep2(f0_step1, voice_range_minimum):
    f0_step2 = copy.deepcopy(f0_step1)
    boundary_list = GetBoundaryList(f0_step1)

    for i in np.arange(1, len(boundary_list) / 2 + 1):
        distance = boundary_list[2 * i -1] - boundary_list[(2 * i) - 2]
        if distance < voice_range_minimum:
            f0_step2[boundary_list[(2 * i) - 2] : boundary_list[2 * i - 1] + 1] = 0
    return f0_step2


####################################################################################################
# Step 3: Voiced sections are extended based on the continuity of F0 contour
def FixStep3(f0_step2, f0_candidates, allowed_range, f0_candidates_score):
    f0_step3 = copy.deepcopy(f0_step2)
    boundary_list = GetBoundaryList(f0_step2)
    multi_channel_f0 = GetMultiChannelF0(f0_step2, boundary_list)
    range = np.zeros((len(boundary_list) / 2, 2))
    threshold1 = 100
    threshold2 = 2200

    count = -1
    for i in np.arange(1, len(boundary_list) / 2 + 1):
        tmp_range = np.zeros(2)
        # Value 100 is optimized.
        extended_f0, tmp_range[1] = ExtendF0(multi_channel_f0[i - 1, :], boundary_list[i * 2 - 1], min(len(f0_step2) - 2,
                                    boundary_list[i * 2 - 1] + threshold1), 1, f0_candidates, allowed_range)
        tmp_f0_sequence, tmp_range[0] = ExtendF0(extended_f0, boundary_list[(i * 2) - 2],
            max(1, boundary_list[(i * 2) - 2] - threshold1), -1, f0_candidates, allowed_range)

        mean_f0 = np.mean(tmp_f0_sequence[tmp_range[0] : tmp_range[1] + 1])
        if threshold2 / mean_f0 < tmp_range[1] - tmp_range[0]:
            count += 1
            multi_channel_f0[count, :] = tmp_f0_sequence
            range[count, :] = tmp_range
    multi_channel_f0 = multi_channel_f0[0 : count + 1, :]
    range = range[0 : count + 1, :]
    if count > -1:
        f0_step3 = MergeF0(multi_channel_f0, range, f0_candidates, f0_candidates_score)
    return f0_step3


####################################################################################################
# Step 4: F0s in short unvoiced section are faked
def FixStep4(f0_step3, threshold):
    f0_step4 = copy.deepcopy(f0_step3)
    boundary_list = GetBoundaryList(f0_step3)

    for i in np.arange(1, len(boundary_list) / 2 ):
        distance = boundary_list[2 * i] - boundary_list[2 * i - 1] - 1
        if distance >= threshold:
            continue
        tmp0 = f0_step3[boundary_list[2 * i - 1]] + 1
        tmp1 = f0_step3[boundary_list[2 * i]] - 1
        c = (tmp1 - tmp0) / (distance + 1)
        count = 1
        for j in np.arange(boundary_list[2 * i - 1] + 1, boundary_list[2 * i]):
            f0_step4[j] = tmp0 + c * count
            count += 1
    return f0_step4


####################################################################################################
def ExtendF0(f0, origin, last_point, shift, f0_candidates, allowed_range):
    threshold = 4
    extended_f0 = copy.deepcopy(f0)
    tmp_f0 = extended_f0[origin]
    shifted_origin = origin

    count = 0
    if shift == 1:
        last_point += 1
    elif shift == -1:
        last_point -= 1
    for i in np.arange(origin, last_point, shift):
        extended_f0[i + shift], _ = SelectBestF0(tmp_f0, f0_candidates[:, i + shift], allowed_range)
        if extended_f0[i + shift] != 0:
            tmp_f0 = extended_f0[i + shift]
            count = 0
            shifted_origin = i + shift
        else:
            count += + 1
        if count == threshold:
            break
    return extended_f0, shifted_origin


####################################################################################################
def GetMultiChannelF0(f0, boundary_list):
    multi_channel_f0 = np.zeros((len(boundary_list) / 2, len(f0)))
    for i in np.arange(1, len(boundary_list) / 2 + 1):
        multi_channel_f0[i - 1, boundary_list[(i * 2) - 2] : boundary_list[i * 2 - 1] + 1] =\
            f0[boundary_list[(i * 2) - 2] : boundary_list[(i * 2) - 1] + 1]
    return multi_channel_f0


####################################################################################################
def MergeF0(multi_channel_f0, range, f0_candidates, f0_candidates_score):
    number_of_channels = multi_channel_f0.shape[0]
    sorted_order = np.argsort(range[:, 0], axis=0, kind='quicksort')
    f0 = multi_channel_f0[sorted_order[0], :]

    for i in np.arange(1, number_of_channels):
        # without overlapping
        if range[sorted_order[i], 0] - range[sorted_order[0], 1] > 0:
            f0[range[sorted_order[i], 0] : range[sorted_order[i], 1] + 1] = multi_channel_f0[sorted_order[i],\
                                                        range[sorted_order[i], 0] : range[sorted_order[i], 1] + 1]
            range[sorted_order[0], 0] = range[sorted_order[i], 0]
            range[sorted_order[0], 1] = range[sorted_order[i], 1]
        else: # with overlapping
            f0, range[sorted_order[0], 1] = MergeF0Sub(f0, range[sorted_order[0], 0], range[sorted_order[0], 1],
              multi_channel_f0[sorted_order[i], :], range[sorted_order[i], 0],
              range[sorted_order[i], 1], f0_candidates, f0_candidates_score)
    return f0


####################################################################################################
def MergeF0Sub(f0_1, st1, ed1, f0_2, st2, ed2, f0_candidates, f0_candidates_score):
    merged_f0 = copy.deepcopy(f0_1)

    # Completely overlapping section
    if st1 <= st2 and ed1 >= ed2:
        new_ed = ed1
        return merged_f0, new_ed
    new_ed = ed2

    score1 = 0
    score2 = 0
    for i in np.arange(st2, ed1 + 1):
        score1 = score1 + SerachScore(f0_1[i], f0_candidates[:, i], f0_candidates_score[:, i])
        score2 = score2 + SerachScore(f0_2[i], f0_candidates[:, i], f0_candidates_score[:, i])
    if score1 > score2:
        merged_f0[ed1 : ed2 + 1] = f0_2[ed1 : ed2 + 1]
    else:
        merged_f0[st2 : ed2 + 1] = f0_2[st2 : ed2 + 1]
    return merged_f0, new_ed


####################################################################################################
def SerachScore(f0, f0_candidates, f0_candidates_score):
    score = 0
    for i in range(f0_candidates.shape[0]):
        if f0 == f0_candidates[i] and score < f0_candidates_score[i]:
            score = f0_candidates_score[i]
    return score


####################################################################################################
def GetF0Candidates(negative_zero_cross, positive_zero_cross, peak, dip, temporal_positions):
    # test this one
    usable_channel = max(0, np.size(negative_zero_cross['interval_locations']) - 2) * \
                     max(0, np.size(positive_zero_cross['interval_locations']) - 2) * \
                     max(0, np.size(peak['interval_locations']) - 2) * \
                     max(0, np.size(dip['interval_locations']) - 2)

    interpolated_f0_list = np.zeros((4, np.size(temporal_positions)))

    if usable_channel > 0:
        interpolated_f0_list[0, :] = interp1d(negative_zero_cross['interval_locations'],
                                              negative_zero_cross['interval_based_f0'],
                                              fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[1, :] = interp1d(positive_zero_cross['interval_locations'],
                                              positive_zero_cross['interval_based_f0'],
                                              fill_value='extrapolate')(temporal_positions)

        interpolated_f0_list[2, :] = interp1d(peak['interval_locations'],
                                              peak['interval_based_f0'],
                                              fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[3, :] = interp1d(dip['interval_locations'],
                                              dip['interval_based_f0'],
                                              fill_value='extrapolate')(temporal_positions)

        interpolated_f0 = np.mean(interpolated_f0_list, axis=0)
    else:
        interpolated_f0 = temporal_positions * 0
    return interpolated_f0


###################################################################################################
def SmoothF0(f0):
    b = np.array([0.0078202080334971724, 0.015640416066994345, 0.0078202080334971724])
    a = np.array([1.0, -1.7347257688092754, 0.76600660094326412])

    smoothed_f0 = np.append(np.append(np.zeros(300), f0), np.zeros(300))
    boundary_list = GetBoundaryList(smoothed_f0)
    multi_channel_f0 = GetMultiChannelF0(smoothed_f0, boundary_list)
    for i in np.arange(1, len(boundary_list) / 2 + 1):
        tmp_f0_contour = FilterF0(multi_channel_f0[i - 1, :], boundary_list[i * 2 - 2], boundary_list[i * 2 - 1], b, a)
        smoothed_f0[boundary_list[i * 2 - 2] : boundary_list[i * 2 - 1] + 1] = \
            tmp_f0_contour[boundary_list[i * 2 - 2] : boundary_list[i * 2 - 1] + 1]

    smoothed_f0 = smoothed_f0[300 : len(smoothed_f0) - 300]
    return smoothed_f0


####################################################################################################
def FilterF0(f0_contour, st, ed, b, a):
    smoothed_f0 = copy.deepcopy(f0_contour)
    smoothed_f0[0 : st] = smoothed_f0[st]
    smoothed_f0[ed + 1: ] = smoothed_f0[ed]
    aaa = lfilter(b, a, smoothed_f0, axis=0)
    bbb = lfilter(b, a, aaa[-1 : : -1], axis=0)
    smoothed_f0 = bbb[-1 : : -1]
    smoothed_f0[0 : st] = 0
    smoothed_f0[ed + 1: ] = 0
    return smoothed_f0


####################################################################################################
def nuttall(N):
    t = np.asmatrix(np.arange(N) * 2 * math.pi / (N-1))
    coefs = np.array([0.355768, -0.487396, 0.144232, -0.012604])
    window = coefs @ np.cos(np.matrix([0,1,2,3]).T @ t)
    return np.squeeze(np.asarray(window))


#####################################################################################################
def round_matlab(n):
    '''
    this function works as Matlab round() function
    python round function choose the nearest even number to n, which is different from Matlab round function
    :param n: input number
    :return: rounded n
    '''
    return int(Decimal(n).quantize(0, ROUND_HALF_UP))


#######################################################################################################
# Note: vuv(1) and vuv(end) are set to 0.
def GetBoundaryList(f0):
    vuv = copy.deepcopy(f0)
    vuv[vuv != 0] = 1
    vuv[0] = 0
    vuv[-1] = 0
    diff_vuv = np.diff(vuv)
    boundary_list = np.where(diff_vuv != 0)[0]
    boundary_list[0:: 2] += 1
    return boundary_list

