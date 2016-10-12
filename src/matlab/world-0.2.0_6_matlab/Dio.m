function f0_parameter = Dio(x, fs, option)
% F0 estimation by DIO
% f0_parameter = Dio(x, fs, opt);
% f0_parameter = Dio(x, fs);
%
% Inputs
%   x  : input signal
%   fs : sampling frequency
%   option : user setting (f0_floor (Hz), f0_ceil (Hz), target_fs (Hz)
%            channels_in_octave (ch), allowed_range, and frame_period (ms))
% Output
%   f0_paramter : f0 infromation
%
% Caution: minimum frame_period is 1.
%
% 2014/04/29: First version was released.
% 2015/07/28: Minor modifications were carried out.

% set default parameters or option
if nargin == 2
  [f0_floor, f0_ceil, channels_in_octave, target_fs, frame_period,...
    allowed_range] = SetDefaultParameters([]);
elseif nargin == 3
  [f0_floor, f0_ceil, channels_in_octave, target_fs, frame_period,...
    allowed_range] = SetDefaultParameters(option);
end;

temporal_positions = 0 : frame_period / 1000 : length(x) / fs;
boundary_f0_list = f0_floor * 2.0 .^...
  ((1 : ceil(log2(f0_ceil / f0_floor) * channels_in_octave)) /...
  channels_in_octave);

% down-sampling to target_fs Hz
[y, actual_fs] = CalculateDownsampledSignal(x, fs, target_fs);
% spectrum with low-pass filtering 2011/1/4
y_spectrum = CalculateSpectrum(y, actual_fs, f0_floor);

[raw_f0_candidate, raw_stability] = CalculateCandidateAndStabirity(...
  length(temporal_positions), boundary_f0_list, length(y),...
  temporal_positions, actual_fs, y_spectrum, f0_floor, f0_ceil);

f0_candidates = SortCandidates(raw_f0_candidate, raw_stability);

f0_parameter.f0 = f0_candidates(1, :);
f0_parameter.f0_candidates = f0_candidates;
f0_parameter.raw_f0_candidates = raw_f0_candidate;
f0_parameter.temporal_positions = temporal_positions;
[f0_parameter.f0, f0_parameter.vuv] =...
  FixF0Contour(f0_candidates, frame_period, f0_floor, allowed_range);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f0_floor, f0_ceil, channels_in_octave, target_fs, frame_period,...
  allowed_range] = SetDefaultParameters(option)

f0_floor = 71;
f0_ceil = 800;
channels_in_octave = 2;
target_fs = 4000;
frame_period = 5;
allowed_range = 0.1;
if isempty(option) ~= 1
  if isfield(option, 'f0_floor') == 1;
    f0_floor = option.f0_floor;
  end;
  if isfield(option, 'f0_ceil') == 1;
    f0_ceil = option.f0_ceil;
  end;
  if isfield(option, 'frame_period') == 1;
    frame_period = option.frame_period;
  end;
  if isfield(option, 'channels_in_octave') == 1;
    channels_in_octave = option.channels_in_octave;
  end;
  if isfield(option, 'target_fs') == 1;
    target_fs = option.target_fs;
  end;
  if isfield(option, 'allowed_range') == 1;
    allowed_range = option.allowed_range;
  end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y, actual_fs] = CalculateDownsampledSignal(x, fs, target_fs)
decimation_ratio = round(fs / target_fs);
if fs < target_fs
  y = x(:, 1);
  actual_fs = fs;
else
  y = decimate(x(:, 1), decimation_ratio, 3);
  actual_fs = fs / decimation_ratio;
end;
y = y - mean(y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y_spectrum = CalculateSpectrum(y, fs, lowest_f0)
fft_size = 2 ^ ceil(log2(length(y) + round(fs / lowest_f0 / 2) * 4));
% Low-cut filtering 2014/01/01
cutoff_in_sample = round(fs / 50); % Under 50 Hz will be cutted.
low_cut_filter = hanning(2 * cutoff_in_sample + 1);
low_cut_filter = -low_cut_filter / sum(low_cut_filter);
low_cut_filter(cutoff_in_sample + 1) = low_cut_filter(cutoff_in_sample + 1) + 1;
low_cut_filter = [low_cut_filter ; zeros(fft_size - length(low_cut_filter), 1)];
low_cut_filter =[low_cut_filter(cutoff_in_sample + 1 : end) ;...
  low_cut_filter(1 : cutoff_in_sample)];

y_spectrum = fft(y, fft_size) .* fft(low_cut_filter, fft_size);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [raw_f0_candidate, raw_f0_stability] =...
  CalculateCandidateAndStabirity(number_of_frames, boundary_f0_list,...
  y_length, temporal_positions, actual_fs, y_spectrum, f0_floor, f0_ceil)

raw_f0_candidate = zeros(length(boundary_f0_list), number_of_frames);
raw_f0_stability = zeros(length(boundary_f0_list), number_of_frames);

for i = 1 : length(boundary_f0_list)
  [interpolated_f0, f0_deviations] = CalculateRawEvent(boundary_f0_list(i),...
    actual_fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil);

  raw_f0_stability(i, :) =...
    exp(-(f0_deviations ./ max(0.0000001, interpolated_f0)));
  raw_f0_candidate(i, :) = interpolated_f0;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f0_candidates, f0_candidates_score] = SortCandidates(...
  f0_candidate_map, stability_map)
[number_of_candidates, number_of_frames] = size(f0_candidate_map);
[~, sorted_index] = sort(stability_map, 1, 'descend');
f0_candidates = zeros(number_of_candidates, number_of_frames);
f0_candidates_score = zeros(number_of_candidates, number_of_frames);

for i = 1 : number_of_frames
  f0_candidates(:, i) =...
    f0_candidate_map(sorted_index(1:number_of_candidates,i), i);
  f0_candidates_score(:,i) =...
    stability_map(sorted_index(1:number_of_candidates,i), i);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f0_candidate, f0_deviations] = CalculateRawEvent(boundary_f0,...
  fs, y_spectrum, y_length, temporal_positions, f0_floor, f0_ceil)
half_filter_length = round(fs / boundary_f0 / 2);
low_pass_filter = nuttall(half_filter_length * 4);

[~, index_bias] = max(low_pass_filter);

spectrum_low_pass_filter = fft(low_pass_filter, length(y_spectrum));

filtered_signal = real(ifft(spectrum_low_pass_filter .* y_spectrum));
filtered_signal = filtered_signal(index_bias + (1 : y_length));

% calculate 4 kinds of event
negative_zero_cross = ZeroCrossingEngine(filtered_signal, fs);
positive_zero_cross = ZeroCrossingEngine(-filtered_signal, fs);
peak = ZeroCrossingEngine(diff(filtered_signal), fs);
dip = ZeroCrossingEngine(-diff(filtered_signal), fs);

[f0_candidate, f0_deviations] =...
  GetF0Candidates(negative_zero_cross, positive_zero_cross, peak,...
  dip, temporal_positions);

% remove untrustful candidates
f0_candidate(f0_candidate > boundary_f0) = 0;
f0_candidate(f0_candidate < (boundary_f0 / 2)) = 0;
f0_candidate(f0_candidate > f0_ceil) = 0;
f0_candidate(f0_candidate < f0_floor) = 0;
f0_deviations(f0_candidate == 0) = 100000; % rough safe guard

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [interpolated_f0, f0_deviations] = GetF0Candidates(...
  negative_zero_cross, positive_zero_cross, peak, dip, temporal_positions)

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
  
  interpolated_f0 = mean(interpolated_f0_list);
  f0_deviations = std(interpolated_f0_list);
else
  interpolated_f0 = temporal_positions * 0;
  f0_deviations = temporal_positions * 0 + 1000;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% negative zero crossing: going from positive to negative
function event_struct = ZeroCrossingEngine(x, fs)
negative_going_points =...
  (1 : length(x))' .*...
  (([x(2 : end) ; x(end)] .* x < 0) .* ([x(2 : end) ; x(end)] < x));

edge_list = negative_going_points(negative_going_points > 0);

fine_edge_list = edge_list - x(edge_list) ./ (x(edge_list + 1) - x(edge_list));

event_struct.interval_locations =...
  (fine_edge_list(1 : end - 1) + fine_edge_list(2 : end)) / 2 / fs;
event_struct.interval_based_f0 = fs ./ diff(fine_edge_list);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function window = nuttall(N)
t = (0 : N - 1)' * 2 * pi / (N - 1);
coefs = [0.355768; -0.487396; 0.144232; -0.012604];
window = cos(t * [0 1 2 3]) * coefs;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f0, vuv] = FixF0Contour(f0_candidates, frame_period, f0_floor,...
  allowed_range)
% if abs((f0(n) - f0(n+1)) / f0(n)) exceeds this value,
% f0(n) is not reliable.
% F0 is continuous at least voice_range_minimum (sample)
voice_range_minimum = round(1 / (frame_period / 1000) / f0_floor) * 2 + 1;

f0_step1 = FixStep1(f0_candidates, voice_range_minimum, allowed_range);
f0_step2 = FixStep2(f0_step1, voice_range_minimum);
section_list = CountNumberOfVoicedSections(f0_step2);
f0_step3 = FixStep3(f0_step2, f0_candidates, section_list, allowed_range);

f0_step4 = FixStep4(f0_step3, f0_candidates, section_list, allowed_range);
f0 = f0_step4;
vuv = f0;
vuv(vuv ~= 0) = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step1: rapid change of f0 contour is replaced by 0
function f0_step1 =...
  FixStep1(f0_candidates, voice_range_minimum, allowed_range)
f0_base = f0_candidates(1, :);
f0_base(1 : voice_range_minimum) = 0;
f0_base(end - voice_range_minimum + 1 : end) = 0;

f0_step1 = f0_base;
for i = voice_range_minimum : length(f0_base)
  if abs((f0_base(i) - f0_base(i-1)) / (0.000001 + f0_base(i))) > allowed_range
    f0_step1(i) = 0;
  end
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step2: short-time voiced period (under voice_range_minimum) is replaced by 0
function f0_step2 = FixStep2(f0_step1, voice_range_minimum)
f0_step2 = f0_step1;
for i = (voice_range_minimum - 1) / 2 + 1 :...
    length(f0_step1) - (voice_range_minimum - 1) / 2
  for j = -(voice_range_minimum - 1) / 2 : (voice_range_minimum - 1) / 2
    if f0_step1(i + j) == 0
      f0_step2(i) = 0;
      break;
    end;
  end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step3: short-time voiced period (under voice_range_minimum) is replaced by 0
function f0_step3 = FixStep3(f0_step2, f0_candidates, section_list,...
  allowed_range)
f0_step3 = f0_step2;

for i = 1 : size(section_list, 1)
  if i == size(section_list, 1)
    limit = length(f0_step3) - 1;
  else
    limit = section_list(i + 1, 1);
  end;
  for j = section_list(i, 2) : limit
    f0_step3(j + 1) = SelectBestF0(f0_step3(j), f0_step3(j - 1),...
      f0_candidates(:, j + 1), allowed_range);
    if f0_step3(j + 1) == 0; break; end;
  end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step3: short-time voiced period (under voice_range_minimum) is replaced by 0
function f0_step4 = FixStep4(f0_step3, f0_candidates, section_list,...
  allowed_range)
f0_step4 = f0_step3;

for i = size(section_list, 1) : -1 : 1
  if i == 1
    limit = 2;
  else
    limit = section_list(i - 1, 2);
  end;
  for j = section_list(i, 1) : -1 : limit
    f0_step4(j - 1) = SelectBestF0(f0_step4(j), f0_step4(j + 1),...
      f0_candidates(:, j - 1), allowed_range);
    if f0_step4(j - 1) == 0; break; end;
  end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function best_f0 = SelectBestF0(current_f0, past_f0, candidates, allowed_range)

reference_f0 = (current_f0 * 3 - past_f0) / 2;
minimum_error = abs(reference_f0 - candidates(1));
best_f0 = candidates(1);

for i = 2 : length(candidates)
  current_error = abs(reference_f0 - candidates(i));
  if current_error < minimum_error
    minimum_error = current_error;
    best_f0 = candidates(i);
  end;
end;
if abs(1 - best_f0 / reference_f0) > allowed_range
  best_f0 = 0;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function voiced_section_list = CountNumberOfVoicedSections(f0)
vuv = f0(:);
vuv(vuv ~= 0) = 1;

diff_vuv = diff(vuv);
boundary_list = [0; find(diff_vuv ~= 0); length(vuv) - 1];
first_section = ceil(-0.5 * diff_vuv(boundary_list(2)));
number_of_voiced_sections =...
  floor((length(boundary_list) - (1 - first_section)) / 2);
voiced_section_list = zeros(number_of_voiced_sections, 2);
for i = 1 : number_of_voiced_sections
  voiced_section_list(i, :) =...
    [1 + boundary_list((i - 1) * 2 + 1 + (1 - first_section)),...
    boundary_list((i * 2) + (1 - first_section))];
end;
