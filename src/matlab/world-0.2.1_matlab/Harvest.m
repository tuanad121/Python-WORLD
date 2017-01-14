function f0_parameter = Harvest(x, fs, option)
% F0 estimation by Harvest
% f0_parameter = Harvest(x, fs, option);
% f0_parameter = Harvest(x, fs);
%
% Input
%   x  : input signal
%   fs : sampling frequency
%   option : user can set f0_floor (Hz), f0_ceil (Hz) and frame_period (ms).
%
% Output
%   f0_paramter : f0 infromation
%
% 2016/11/14: A prototype was released.
% 2016/12/18: First version.
% 2016/12/19: Frame period became controllable.
% 2016/12/28: Refactoring

% set parameters
if nargin == 2
  [f0_floor, f0_ceil, frame_period] = SetDefaultParameters([]);
elseif nargin == 3
  [f0_floor, f0_ceil, frame_period] = SetDefaultParameters(option);
end;
% You can control the frame period. But, the estimation is carried out with
% 1 ms frame shift.
basic_frame_period = 1;
target_fs = 8000;

basic_temporal_positions = 0 : basic_frame_period / 1000 : length(x) / fs;

channels_in_octave = 40;
% Frequency range is expanded to accurately obtain the F0 candidates
adjusted_f0_floor = f0_floor * 0.9;
adjusted_f0_ceil = f0_ceil * 1.1;
boundary_f0_list = adjusted_f0_floor * 2.0 .^...
  ((1 : ceil(log2(adjusted_f0_ceil / adjusted_f0_floor) *...
  channels_in_octave)) / channels_in_octave);

% down-sampling to target_fs Hz
[y, actual_fs] = GetDownsampledSignal(x, fs, target_fs);
fft_size = 2 ^ ceil(log2(length(y) + round(fs / f0_floor * 4) + 1));
y_spectrum = fft(y, fft_size);

raw_f0_candidates = GetRawF0Candidates(...
  length(basic_temporal_positions), boundary_f0_list, length(y),...
  basic_temporal_positions, actual_fs, y_spectrum, f0_floor, f0_ceil);

[f0_candidates, number_of_candidates] =...
  DetectOfficialF0Candidates(raw_f0_candidates);
f0_candidates = OverlapF0Candidates(f0_candidates, number_of_candidates);

[f0_candidates, f0_scores] = RefineCandidates(y, actual_fs,...
  basic_temporal_positions, f0_candidates, f0_floor, f0_ceil);

[f0_candidates, f0_scores] =...
  RemoveUnreliableCandidates(f0_candidates, f0_scores);

[connected_f0, vuv] = FixF0Contour(f0_candidates, f0_scores);
smoothed_f0 = SmoothF0Contour(connected_f0);

temporal_positions = 0 : frame_period / 1000 : length(x) / fs;
f0_parameter.temporal_positions = temporal_positions;
f0_parameter.f0 = smoothed_f0(min(length(smoothed_f0) - 1,...
  round(temporal_positions * 1000)) + 1);
f0_parameter.vuv =...
  vuv(min(length(smoothed_f0) - 1, round(temporal_positions * 1000)) + 1);
f0_parameter.f0_candidates = f0_candidates;
% f0_parameter.f0_candidates_score = f0_candidates_score;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [new_f0_candidates, new_f0_scores] =...
  RemoveUnreliableCandidates(f0_candidates, f0_scores)
new_f0_candidates = f0_candidates;
new_f0_scores = f0_scores;
threshold = 0.05;

f0_length = size(f0_candidates, 2);
number_of_candidates = size(f0_candidates, 1);

for i = 2 : f0_length - 1
  for j = 1 : number_of_candidates
    reference_f0 = f0_candidates(j, i);
    if reference_f0 == 0; continue; end;
    [~, min_error1] = SelectBestF0(reference_f0, f0_candidates(:, i + 1), 1);
    [~, min_error2] = SelectBestF0(reference_f0, f0_candidates(:, i - 1), 1);
    min_error = min([min_error1, min_error2]);
    if min_error > threshold
      new_f0_candidates(j, i) = 0;
      new_f0_scores(j, i) = 0;
    end;
  end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function new_f0_candidates = OverlapF0Candidates(f0_candidates, max_candidates)
n = 3; % This is the optimzied parameter.

number_of_candidates = n * 2 + 1;
new_f0_candidates = f0_candidates(number_of_candidates, :);
for i = 0 : number_of_candidates - 1
  st = max(-(i - n) + 1, 1);
  ed = min(-(i - n), 0);
  new_f0_candidates((1 : max_candidates) + i * max_candidates, st : end + ed) =...
    f0_candidates(1 : max_candidates, -ed + 1 : end - (st - 1));
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y, actual_fs] = GetDownsampledSignal(x, fs, target_fs)
decimation_ratio = round(fs / target_fs);

offset = ceil(140 / decimation_ratio) * decimation_ratio;
x = [ones(offset, 1) * x(1); x(:); ones(offset, 1) * x(end)];

if fs < target_fs
  y = x(:, 1);
  actual_fs = fs;
else
  y0 = decimate(x(:, 1), decimation_ratio, 3);
  actual_fs = fs / decimation_ratio;
  y = y0(offset / decimation_ratio + 1 : end - offset / decimation_ratio);
end;
y = y - mean(y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function raw_f0_candidates = GetRawF0Candidates(number_of_frames,...
  boundary_f0_list, y_length, temporal_positions, actual_fs, y_spectrum,...
  f0_floor, f0_ceil)
raw_f0_candidates = zeros(length(boundary_f0_list), number_of_frames);

for i = 1 : length(boundary_f0_list)
  raw_f0_candidates(i, :) = GetF0CandidateFromRawEvent(boundary_f0_list(i),...
    actual_fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f0_candidates, number_of_candidates] =...
  DetectOfficialF0Candidates(raw_f0_candidates)
[number_of_channels, number_of_frames] = size(raw_f0_candidates);
f0_candidates = zeros(round(number_of_channels / 10), number_of_frames);
number_of_candidates = 0;
threshold = 10;

for i = 1 : number_of_frames
  tmp = raw_f0_candidates(:, i);
  tmp(tmp > 0) = 1;
  tmp(1) = 0;
  tmp(end) = 0;
  tmp = diff(tmp);
  st = find(tmp == 1);
  ed = find(tmp == -1);
  count = 1;
  for j = 1 : length(st)
    dif = ed(j) - st(j);
    if dif >= threshold
      tmp_f0 = raw_f0_candidates(st(j) + 1: ed(j), i);
      f0_candidates(count, i) = mean(tmp_f0);
      count = count + 1;
    end;
  end;
  number_of_candidates = max(number_of_candidates, count - 1);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f0_candidate = GetF0CandidateFromRawEvent(boundary_f0,...
  fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil)
filter_length_half = round(fs / boundary_f0 * 2);
band_pass_filter_base = nuttall(filter_length_half * 2 + 1);
shifter =...
  cos(2 * pi * boundary_f0 * (-filter_length_half : filter_length_half)' / fs);
band_pass_filter = band_pass_filter_base(:) .* shifter(:);

index_bias = filter_length_half + 1;
spectrum_low_pass_filter = fft(band_pass_filter, length(y_spectrum));

filtered_signal = real(ifft(spectrum_low_pass_filter .* y_spectrum));
filtered_signal = filtered_signal(index_bias + (1 : y_length));

% calculate 4 kinds of event
negative_zero_cross = ZeroCrossingEngine(filtered_signal, fs);
positive_zero_cross = ZeroCrossingEngine(-filtered_signal, fs);
peak = ZeroCrossingEngine(diff(filtered_signal), fs);
dip = ZeroCrossingEngine(-diff(filtered_signal), fs);

f0_candidate = GetF0CandidateContour(negative_zero_cross,...
  positive_zero_cross, peak, dip, temporal_positions);

% remove unreliable candidates
% 1.1 and 0.9 are the fixed parameters.
f0_candidate(f0_candidate > boundary_f0 * 1.1) = 0;
f0_candidate(f0_candidate < boundary_f0 * 0.9) = 0;
f0_candidate(f0_candidate > f0_ceil) = 0;
f0_candidate(f0_candidate < f0_floor) = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f0_candidate = GetF0CandidateContour(negative_zero_cross,...
  positive_zero_cross, peak, dip, temporal_positions)
usable_channel =...
  max(0, length(negative_zero_cross.interval_locations) - 2) *...
  max(0, length(positive_zero_cross.interval_locations) - 2) *...
  max(0, length(peak.interval_locations) - 2) *...
  max(0, length(dip.interval_locations) - 2);

if usable_channel > 0
  interpolated_f0_list = zeros(4, length(temporal_positions));
  interpolated_f0_list(1, :) =...
    interp1(negative_zero_cross.interval_locations,...
    negative_zero_cross.interval_based_f0, temporal_positions,...
    'linear', 'extrap');
  interpolated_f0_list(2, :) =...
    interp1(positive_zero_cross.interval_locations,...
    positive_zero_cross.interval_based_f0, temporal_positions,...
    'linear', 'extrap');
  interpolated_f0_list(3, :) = interp1(peak.interval_locations,...
    peak.interval_based_f0, temporal_positions, 'linear', 'extrap');
  interpolated_f0_list(4, :) = interp1(dip.interval_locations,...
    dip.interval_based_f0, temporal_positions, 'linear', 'extrap');
  
  f0_candidate = mean(interpolated_f0_list);
else
  f0_candidate = temporal_positions * 0;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% negative zero crossing: going from positive to negative
function event_struct = ZeroCrossingEngine(x, fs)
negative_going_points = (1 : length(x))' .*...
  (([x(2 : end) ; x(end)] .* x < 0) .* ([x(2 : end) ; x(end)] < x));

edge_list = negative_going_points(negative_going_points > 0);
fine_edge_list = edge_list - x(edge_list) ./ (x(edge_list + 1) - x(edge_list));
event_struct.interval_locations =...
  (fine_edge_list(1 : end - 1) + fine_edge_list(2 : end)) / 2 / fs;
event_struct.interval_based_f0 = fs ./ diff(fine_edge_list);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f0, vuv] = FixF0Contour(f0_candidates, f0_scores)
f0_base = SearchF0Base(f0_candidates, f0_scores);
f0_step1 = FixStep1(f0_base, 0.008); % optimized
f0_step2 = FixStep2(f0_step1, 6); % optimized
f0_step3 = FixStep3(f0_step2, f0_candidates, 0.18, f0_scores); % optimized
f0 = FixStep4(f0_step3, 9); % optimized
vuv = f0;
vuv(vuv ~= 0) = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F0s with the highest score are selected as a basic f0 contour.
function f0_base = SearchF0Base(f0_candidates, f0_scores)
f0_base = f0_candidates(1, :) * 0;
for i = 1 : length(f0_base)
  [~, max_index] = max(f0_scores(:, i));
  f0_base(i) = f0_candidates(max_index, i);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: Rapid change of f0 contour is replaced by 0
function f0_step1 = FixStep1(f0_base, allowed_range)
f0_step1 = f0_base;
f0_step1(1) = 0;
f0_step1(2) = 0;

for i = 3 : length(f0_base)
  if f0_base(i) == 0; continue; end;
  reference_f0 = f0_base(i - 1) * 2 - f0_base(i - 2);
  if abs((f0_base(i) - reference_f0) / reference_f0) > allowed_range &&...
      abs((f0_base(i) - f0_base(i - 1)) / f0_base(i - 1)) > allowed_range
    f0_step1(i) = 0;
  end
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 2: Voiced sections with a short period are removed
function f0_step2 = FixStep2(f0_step1, voice_range_minimum)
f0_step2 = f0_step1;
boundary_list = GetBoundaryList(f0_step1);

for i = 1 : length(boundary_list) / 2
  distance = boundary_list(2 * i) - boundary_list((2 * i) - 1);
  if distance < voice_range_minimum
    f0_step2(boundary_list((2 * i) - 1) : boundary_list(2 * i)) = 0;
  end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 3: Voiced sections are extended based on the continuity of F0 contour
function f0_step3 =...
  FixStep3(f0_step2, f0_candidates, allowed_range, f0_scores)
f0_step3 = f0_step2;
boundary_list = GetBoundaryList(f0_step2);
multi_channel_f0 = GetMultiChannelF0(f0_step2, boundary_list);
range = zeros(length(boundary_list) / 2, 2);
threshold1 = 100;
threshold2 = 2200;

count = 0;
for i = 1 : length(boundary_list) / 2
  [extended_f0, tmp_range(2)] = ExtendF0(multi_channel_f0(i, :),...
    boundary_list(i * 2), min(length(f0_step2) - 1,...
    boundary_list(i * 2) + threshold1), 1, f0_candidates, allowed_range);
  [tmp_f0_sequence, tmp_range(1)] =...
    ExtendF0(extended_f0, boundary_list((i * 2) - 1),...
    max(2, boundary_list((i * 2) - 1) - threshold1), -1, f0_candidates,...
    allowed_range);

  mean_f0 = mean(tmp_f0_sequence(tmp_range(1) : tmp_range(2)));
  if threshold2 / mean_f0 < tmp_range(2) - tmp_range(1)
    count = count + 1;
    multi_channel_f0(count, :) = tmp_f0_sequence;
    range(count, :) = tmp_range;
  end;
end;
multi_channel_f0 = multi_channel_f0(1 : count, :);
range = range(1 : count, :);

if count > 0
  f0_step3 =...
    MergeF0(multi_channel_f0, range, f0_candidates, f0_scores);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function multi_channel_f0 = GetMultiChannelF0(f0, boundary_list)
multi_channel_f0 = zeros(length(boundary_list) / 2, length(f0));
for i = 1 : length(boundary_list) / 2
  multi_channel_f0(i, boundary_list((i * 2) - 1) : boundary_list(i * 2)) =...
    f0(boundary_list((i * 2) - 1) : boundary_list((i * 2)));
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f0 =...
  MergeF0(multi_channel_f0, range, f0_candidates, f0_scores)
number_of_channels = size(multi_channel_f0, 1);
[~, sorted_order] = sort(range(:, 1));
f0 = multi_channel_f0(sorted_order(1), :);

for i = 2 : number_of_channels
  % without overlapping
  if range(sorted_order(i), 1) - range(sorted_order(1), 2) > 0
    f0(range(sorted_order(i), 1) : range(sorted_order(i), 2)) =...
      multi_channel_f0(sorted_order(i),...
      range(sorted_order(i), 1) : range(sorted_order(i), 2));
    range(sorted_order(1), 1) = range(sorted_order(i), 1);
    range(sorted_order(1), 2) = range(sorted_order(i), 2);
  else % with overlapping
    [f0, range(sorted_order(1), 2)] =...
      MergeF0Sub(f0, range(sorted_order(1), 1), range(sorted_order(1), 2),...
      multi_channel_f0(sorted_order(i), :), range(sorted_order(i), 1),...
      range(sorted_order(i), 2), f0_candidates, f0_scores);
  end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Two F0 contours (f0_1 and F0_2) are merged.
% F0_1 has the voiced section from st1 to ed1.
function [merged_f0, new_ed] = MergeF0Sub(f0_1, st1, ed1, f0_2, st2, ed2,...
  f0_candidates, f0_scores)
merged_f0 = f0_1;

% Completely overlapping section
if st1 <= st2 && ed1 >= ed2
  new_ed = ed1;
  return;
end;
new_ed = ed2;

score1 = 0;
score2 = 0;
for i = st2 : ed1
  score1 = score1 +...
    SerachScore(f0_1(i), f0_candidates(:, i), f0_scores(:, i));
  score2 = score2 +...
    SerachScore(f0_2(i), f0_candidates(:, i), f0_scores(:, i));
end;
if score1 > score2
  merged_f0(ed1 : ed2) = f0_2(ed1 : ed2);
else
  merged_f0(st2 : ed2) = f0_2(st2 : ed2);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function score = SerachScore(f0, f0_candidates, f0_scores)
score = 0;
for i = 1 : length(f0_candidates)
  if f0 == f0_candidates(i) && score < f0_scores(i)
    score = f0_scores(i);
  end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [extended_f0, shifted_origin] = ExtendF0(f0, origin, last_point,...
  shift, f0_candidates, allowed_range)
threshold = 4;
extended_f0 = f0;
tmp_f0 = extended_f0(origin);
shifted_origin = origin;

count = 0;
for i = origin : shift : last_point
  extended_f0(i + shift) =...
    SelectBestF0(tmp_f0, f0_candidates(:, i + shift), allowed_range);
  if extended_f0(i + shift) ~= 0
    tmp_f0 = extended_f0(i + shift);
    count = 0;
    shifted_origin = i + shift;
  else
    count = count + 1;
  end;
  if count == threshold; break; end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [best_f0, best_error] =...
  SelectBestF0(reference_f0, f0_candidates, allowed_range)
best_f0 = 0;
best_error = allowed_range;

for i = 1 : length(f0_candidates)
  tmp = abs(reference_f0 - f0_candidates(i)) / reference_f0;
  if tmp > best_error; continue; end;
  best_f0 = f0_candidates(i);
  best_error = tmp;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 4: F0s in short unvoiced section are faked
function f0_step4 = FixStep4(f0_step3, threshold)
f0_step4 = f0_step3;
boundary_list = GetBoundaryList(f0_step3);

for i = 1 : length(boundary_list) / 2 - 1
  distance = boundary_list((2 * i) + 1) - boundary_list(2 * i) - 1;
  if distance >= threshold; continue; end;
  boundary0 = f0_step3(boundary_list(2 * i)) + 1; % vuv from 0 -> 1
  boundary1 = f0_step3(boundary_list((2 * i) + 1)) - 1; % vuv from 1 -> 0
  coefficient = (boundary1 - boundary0) / (distance + 1);
  count = 1;
  for j = boundary_list(2 * i) + 1 : boundary_list((2 * i) + 1) - 1
    f0_step4(j) = boundary0 + coefficient * count;
    count = count + 1;
  end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [new_f0_candidates, f0_scores] = RefineCandidates(x, fs,...
  temporal_positions, f0_candidates, f0_floor, f0_ceil)
new_f0_candidates = f0_candidates;
f0_scores = f0_candidates * 0;

for i = 1 : length(temporal_positions)
  for j = 1 : size(f0_candidates, 1)
    tmp_f0 = f0_candidates(j, i);
    if tmp_f0 == 0; continue; end;
    [new_f0_candidates(j, i), f0_scores(j, i)] =...
      GetRefinedF0(x, fs, temporal_positions(i), tmp_f0, f0_floor, f0_ceil);
  end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [refined_f0, refined_score] =...
  GetRefinedF0(x, fs, current_time, current_f0, f0_floor, f0_ceil)
half_window_length = ceil(3 * fs / current_f0 / 2);
window_length_in_time = (2 * half_window_length + 1) / fs;
base_time = (-half_window_length : half_window_length)' / fs;
fft_size = 2 ^ ceil(log2((half_window_length * 2 + 1)) + 1);
frequency_axis = ((0 : fft_size - 1) / fft_size * fs)';

% First-aid treatment
base_index = round((current_time + base_time) * fs + 0.001);
index_time = (base_index - 1) / fs;
window_time = index_time - current_time;
main_window =...
  0.42 + 0.5 * cos(2 * pi * window_time / window_length_in_time) +...
  0.08 * cos(4 * pi * window_time / window_length_in_time);
diff_window = -(diff([0; main_window]) + diff([main_window; 0])) / 2;

safe_index = max(1, min(length(x), base_index));
spectrum = fft(x(safe_index) .* main_window, fft_size);
diff_spectrum = fft(x(safe_index) .* diff_window, fft_size);
numerator_i = real(spectrum) .* imag(diff_spectrum) -...
  imag(spectrum) .* real(diff_spectrum);
power_spectrum = abs(spectrum) .^ 2;
instantaneous_frequency =...
  frequency_axis + numerator_i ./ power_spectrum * fs / 2 / pi;

number_of_harmonics = min(floor(fs / 2 / current_f0), 6); % with safe guard
harmonics_index = (1 : number_of_harmonics)';
index_list = round(current_f0 * fft_size / fs * harmonics_index) + 1;
instantaneous_frequency_list = instantaneous_frequency(index_list);
amplitude_list = sqrt(power_spectrum(index_list));
refined_f0 = sum(amplitude_list .* instantaneous_frequency_list) /...
  sum(amplitude_list .* harmonics_index);

variation = abs(((instantaneous_frequency_list ./ harmonics_index) -...
  current_f0) ./ current_f0);
refined_score = 1 / (0.000000000001 + mean(variation));
if refined_f0 < f0_floor || refined_f0 > f0_ceil || refined_score < 2.5
  refined_f0 = 0;
  refined_score = 0;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function smoothed_f0 = SmoothF0Contour(f0)
b = [0.0078202080334971724, 0.015640416066994345, 0.0078202080334971724];
a = [1.0, -1.7347257688092754, 0.76600660094326412];

smoothed_f0 = [zeros(1, 300), f0, zeros(1, 300)];
boundary_list = GetBoundaryList(smoothed_f0);
multi_channel_f0 = GetMultiChannelF0(smoothed_f0, boundary_list);
for i = 1 : length(boundary_list) / 2
  tmp_f0_contour = FilterF0Contour(multi_channel_f0(i, :),...
    boundary_list((i * 2) - 1), boundary_list(i * 2), b, a);
  smoothed_f0(boundary_list((i * 2) - 1) : boundary_list(i * 2)) =...
    tmp_f0_contour(boundary_list((i * 2) - 1) : boundary_list(i * 2));
end;
smoothed_f0 = smoothed_f0(301 : end - 300);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function smoothed_f0 = FilterF0Contour(f0, st, ed, b, a)
smoothed_f0 = f0;
smoothed_f0(1 : st - 1) = smoothed_f0(st);
smoothed_f0(ed + 1: end) = smoothed_f0(ed);
aaa = filter(b, a, smoothed_f0);
bbb = filter(b, a, aaa(end : -1 : 1));
smoothed_f0 = bbb(end : -1 : 1);
smoothed_f0(1 : st - 1) = 0;
smoothed_f0(ed + 1: end) = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function window = nuttall(N)
t = (0 : N - 1)' * 2 * pi / (N - 1);
coefs = [0.355768; -0.487396; 0.144232; -0.012604];
window = cos(t * [0 1 2 3]) * coefs;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: vuv(1) and vuv(end) are set to 0.
function boundary_list = GetBoundaryList(f0)
vuv = f0;
vuv(vuv ~= 0) = 1;
vuv(1) = 0;
vuv(end) = 0;
diff_vuv = diff(vuv);
boundary_list = find(diff_vuv ~= 0);
boundary_list(1 : 2 : end) = boundary_list(1 : 2 : end) + 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f0_floor, f0_ceil, frame_period] = SetDefaultParameters(option)
f0_floor = 71;
f0_ceil = 800;
frame_period = 5;
if isempty(option) ~= 1
  if isfield(option, 'f0_floor') == 1
    f0_floor = option.f0_floor;
  end;
  if isfield(option, 'f0_ceil') == 1
    f0_ceil = option.f0_ceil;
  end;
  if isfield(option, 'frame_period') == 1
    frame_period = option.frame_period;
  end;
end;
