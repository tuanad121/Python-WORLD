function source_object = D4C(x, fs, f0_object, option)
% Band-aperiodicity estimation based on D4C
% source_object = D4C(x, fs, f0_object, option);
% source_object = D4C(x, fs, f0_object);
%
% Input
%   x  : input signal
%   fs : sampling frequency
%   f0_object : F0 information object
%   option    : It has two parameters (threshold and fft_size)
%               Parameter threshold is used for D4C Love Train (from 0 to 1).
%               Default parameter is 0.85.
%               You can use conventional D4C by setting the parameter to 0.
%               Parameter fft_size is used when you modified the fft_size
%               in CheapTrick(). The same value is required to synthesize
%               the waveform.
%
% Output
%   source_object : estimated band-aperiodicity.
%
% 2015/05/29 : First version was released.
% 2016/12/26 : D4C Love Train is implemented.
% 2016/12/28 : Refactoring
% 2016/12/31 : Minor bugs and a lengthy processing were fixed.
% 2017/01/02 : Minor bugs were fixed.

% set default parameters
f0_low_limit = 47;
fft_size = 2 ^ ceil(log2(4 * fs / f0_low_limit + 1));
% The size of aperiodicity must be the same as that of spectrogram.
f0_low_limit_for_spectrum = 71;
fft_size_for_spectrum =...
  2 ^ ceil(log2(3 * fs / f0_low_limit_for_spectrum + 1));
threshold = 0.85;
if nargin == 4
  if isfield(option, 'threshold') == 1
    threshold = option.threshold;
  end;
  if isfield(option, 'fft_size') == 1
    fft_size_for_spectrum = option.fft_size;
  end;
end;
upper_limit = 15000;
frequency_interval = 3000;

source_object = f0_object;

temporal_positions = f0_object.temporal_positions;
f0 = f0_object.f0;
if isfield(f0_object, 'vuv')
  f0(f0_object.vuv == 0) = 0;
end;

number_of_aperiodicities =...
  floor(min(upper_limit, fs / 2 - frequency_interval) / frequency_interval);

% The window function used for the CalculateFeature() is designed here to
% speed up
window_length = floor(frequency_interval / (fs / fft_size)) * 2 + 1;
window = nuttall(window_length);

aperiodicity = zeros(fft_size_for_spectrum / 2 + 1, length(f0));
ap_debug = zeros(number_of_aperiodicities, length(f0));

frequency_axis = (0 : fft_size_for_spectrum / 2) * fs / fft_size_for_spectrum;
coarse_axis = [(0 : number_of_aperiodicities) * frequency_interval, fs / 2]';

for i = 1 : length(f0)
  if D4CLoveTrain(x, fs, f0(i), temporal_positions(i), threshold) == 0
    aperiodicity(:, i) = 1 - 0.000000000001;
    continue;
  end;
  current_f0 =  max(f0_low_limit, f0(i));
  coarse_aperiodicity = EstimateOneSlice(x, fs, current_f0,...
    frequency_interval, temporal_positions(i), fft_size,...
    number_of_aperiodicities, window);
  coarse_aperiodicity =...
    max(0, coarse_aperiodicity - (current_f0 - 100) * 2 / 100);
  ap_debug(:, i) = -coarse_aperiodicity; % for debug;
  
  aperiodicity(:, i) = 10 .^...
    (interp1(coarse_axis, [-60; -coarse_aperiodicity(:); -0.000000000001],...
    frequency_axis, 'linear') / 20);
end;

source_object.aperiodicity = aperiodicity;
source_object.coarse_ap = ap_debug;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function vuv = D4CLoveTrain(x, fs, current_f0, current_position, threshold)
vuv = 0;
if current_f0 == 0
  return;
end;

lowest_f0 = 40;
current_f0 = max(current_f0, lowest_f0);
fft_size = 2 ^ ceil(log2(3 * fs / lowest_f0 + 1));
% Cumulative powers at 100, 4000, 7900 Hz are used for VUV identification.
boundary0 = ceil(100 / (fs / fft_size)) + 1;
boundary1 = ceil(4000 / (fs / fft_size)) + 1;
boundary2 = ceil(7900 / (fs / fft_size)) + 1;

waveform =...
  GetWindowedWaveform(x, fs, current_f0, current_position, 1.5, 2);
power_spectrum = abs(fft(waveform, fft_size)) .^ 2;
power_spectrum(1 : boundary0) = 0.0;
cumlative_spectrum = cumsum(power_spectrum);

if cumlative_spectrum(boundary1) / cumlative_spectrum(boundary2) > threshold
  vuv = 1;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function coarse_aperiodicity = EstimateOneSlice(x, fs, current_f0,...
  frequency_interval, current_position, fft_size, number_of_aperiodicities,...
  window)
if current_f0 == 0
  coarse_aperiodicity = zeros(number_of_aperiodicities, 1);
  return;
end;

static_centroid =...
  GetStaticCentroid(x, fs, current_f0, current_position, fft_size);
waveform = GetWindowedWaveform(x, fs, current_f0, current_position, 2, 1);
smoothed_power_spectrum =...
  GetSmoothedPowerSpectrum(waveform, fs, current_f0, fft_size);
static_group_delay =...
  GetStaticGroupDelay(static_centroid, smoothed_power_spectrum, fs,...
  current_f0, fft_size);
coarse_aperiodicity =...
  GetCoarseAperiodicity(static_group_delay, fs, fft_size,...
  frequency_interval, number_of_aperiodicities, window);
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function waveform = GetWindowedWaveform(x, fs, current_f0, current_position,...
  half_length, window_type) % 1: hanning, 2: blackman
%  prepare internal variables
half_window_length = round(half_length * fs / current_f0);
base_index = (-half_window_length : half_window_length)';
index = round(current_position * fs + 0.001) + 1 + base_index;
safe_index = min(length(x), max(1, round(index)));

%  wave segments and set of windows preparation
segment = x(safe_index);
time_axis = base_index / fs / half_length;
if window_type == 1 % hanning
  window = 0.5 * cos(pi * time_axis * current_f0) + 0.5;
else % blackman
  window = 0.08 * cos(pi * time_axis * current_f0 * 2) +...
    0.5 * cos(pi * time_axis * current_f0) + 0.42;
end;
waveform = segment .* window - window * mean(segment .* window) / mean(window);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function spectral_envelope = GetSmoothedPowerSpectrum(waveform, fs, f0,...
  fft_size)
power_spectrum = abs(fft(waveform, fft_size)) .^ 2;
spectral_envelope = DCCorrection(power_spectrum, fs, fft_size, f0);
spectral_envelope = LinearSmoothing(spectral_envelope, fs, fft_size, f0);
spectral_envelope = [spectral_envelope; spectral_envelope(end - 1 : -1 : 2)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function centroid = GetStaticCentroid(x, fs, current_f0, current_position,...
    fft_size)
waveform1 = GetWindowedWaveform(x, fs, current_f0,...
  current_position + 1 / current_f0 / 4, 2, 2);
waveform2 = GetWindowedWaveform(x, fs, current_f0,...
  current_position - 1 / current_f0 / 4, 2, 2);
centroid1 = GetCentroid(waveform1, fft_size);
centroid2 = GetCentroid(waveform2, fft_size);
centroid = DCCorrection(centroid1 + centroid2, fs, fft_size, current_f0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function centroid = GetCentroid(x, fft_size)
time_axis = (1 : length(x))';
x = x(:) ./ sqrt(sum(x.^2));

% Centroid calculation on frequency domain.
spectrum = fft(x, fft_size);
weighted_spectrum = fft(-x .* time_axis * 1i, fft_size);
centroid = -imag(weighted_spectrum) .* real(spectrum) +...
  imag(spectrum) .* real(weighted_spectrum);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function group_delay = GetStaticGroupDelay(static_centroid,...
  smoothed_power_spectrum, fs, f0, fft_size)
group_delay = static_centroid ./ smoothed_power_spectrum;
group_delay = LinearSmoothing(group_delay, fs, fft_size, f0 / 2);
group_delay = [group_delay; group_delay(end - 1 : -1 : 2)];
smoothed_group_delay = LinearSmoothing(group_delay, fs, fft_size, f0);
group_delay = group_delay(1 : fft_size / 2 + 1) - smoothed_group_delay;
group_delay = [group_delay; group_delay(end - 1 : -1 : 2)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function smoothed_group_delay =...
  LinearSmoothing(group_delay, fs, fft_size, width)
double_frequency_axis = (0 : 2 * fft_size - 1)' / fft_size * fs - fs;
double_spectrum = [group_delay; group_delay];

double_segment = cumsum(double_spectrum * (fs / fft_size));
center_frequency = (0 : fft_size / 2)' / fft_size * fs;
low_levels = interp1H(double_frequency_axis + fs / fft_size / 2,...
  double_segment, center_frequency - width / 2);
high_levels = interp1H(double_frequency_axis + fs / fft_size / 2,...
  double_segment, center_frequency + width / 2);

smoothed_group_delay = (high_levels - low_levels) / width;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function signal = DCCorrection(signal, fs, fft_size, f0)
frequency_axis = (0 : fft_size - 1)' / fft_size * fs;
low_frequency_axis = frequency_axis(frequency_axis <  f0 + fs / fft_size);
low_frequency_replica = interp1(f0 - low_frequency_axis,...
  signal(frequency_axis < f0 + fs / fft_size),...
  low_frequency_axis(:), 'linear', 'extrap');

signal(frequency_axis < f0) =...
  low_frequency_replica(frequency_axis < f0) + signal(frequency_axis < f0);

signal(end : -1 : fft_size / 2 + 2) = signal(2 : fft_size / 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function coarse_aperiodicity = GetCoarseAperiodicity(group_delay, fs,...
  fft_size, frequency_interval, number_of_aperiodicities, window)
boundary = round(fft_size / length(window) * 8);

half_window_length = floor(length(window) / 2);
coarse_aperiodicity = zeros(number_of_aperiodicities, 1);
for i = 1 : number_of_aperiodicities
  center = floor(frequency_interval * i / (fs / fft_size));
  segment = group_delay((center - half_window_length :...
    center + half_window_length) + 1) .* window;
  power_spectrum = abs(fft(segment, fft_size)) .^ 2;
  
  cumulative_power_spectrum =...
    cumsum(sort(power_spectrum(1 : fft_size / 2 + 1)));
  coarse_aperiodicity(i) =...
    -10 * log10(cumulative_power_spectrum(fft_size / 2 - boundary) /...
    cumulative_power_spectrum(end));
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function window = nuttall(N)
t = (0 : N - 1)' * 2 * pi / (N - 1);
coefs = [0.355768; -0.487396; 0.144232; -0.012604];
window = cos(t * [0 1 2 3]) * coefs;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the imprementation of a matlab function
function yi = interp1H(x, y, xi)
delta_x = x(2) - x(1);
xi = max(x(1), min(x(end), xi));
xi_base = floor((xi - x(1)) / delta_x);
xi_fraction = (xi - x(1)) / delta_x - xi_base;
delta_y = [diff(y); 0];
yi = y(xi_base + 1) + delta_y(xi_base + 1) .* xi_fraction;
