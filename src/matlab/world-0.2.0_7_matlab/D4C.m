function source_object = D4C(x, fs, f0_object)
% Band-aperiodicity estimation based on D4C
% source_object = D4C(x, fs, f0_object);
%
% Inputs
%   x  : input signal
%   fs : sampling frequency
%   f0_object : F0 information object
% Output
%   source_object : estimated band-aperiodicity.
%
% 2015/5/29: First version was released.

% set default parameters
f0_low_limit = 71;
fft_size = 2 ^ ceil(log2(4 * fs / f0_low_limit + 1));
fft_size_for_spectrum = 2 ^ ceil(log2(3 * fs / f0_low_limit + 1));
upper_limit = 15000;
frequency_interval = 3000;

source_object = f0_object;

temporal_positions = f0_object.temporal_positions;
f0_sequence = f0_object.f0;
if isfield(f0_object, 'vuv')
  f0_sequence(f0_object.vuv == 0) = 0;
end;

number_of_aperiodicity =...
  floor(min(upper_limit, fs / 2 - frequency_interval) / frequency_interval);

% The window function used for the CalculateFeature() is designed here to
% speed up
window_length = floor(frequency_interval / (fs / fft_size)) * 2 + 1;
window = nuttall(window_length);

aperiodicity = zeros(fft_size_for_spectrum / 2 + 1, length(f0_sequence));
ap_debug = zeros(number_of_aperiodicity, length(f0_sequence));

frequency_axis = (0 : fft_size_for_spectrum / 2) * fs / fft_size_for_spectrum;
coarse_axis = [(0 : number_of_aperiodicity) * frequency_interval, fs / 2]';

for i = 1 : length(f0_sequence);
  if f0_sequence(i) == 0; aperiodicity(:, i) = 0; continue; end;
  coarse_aperiodicity = EstimateOneSlice(x, fs, f0_sequence(i),...
    frequency_interval, temporal_positions(i), fft_size,...
    number_of_aperiodicity, window);
  coarse_aperiodicity =...
    max(0, coarse_aperiodicity - (f0_sequence(i) - 100) * 2 / 100);
  ap_debug(:, i) = coarse_aperiodicity; % for debug;
  aperiodicity(:, i) = 10 .^...
    (interp1(coarse_axis, [-60; -coarse_aperiodicity(:); 0],...
    frequency_axis, 'linear') / 20);
end;

source_object.aperiodicity = aperiodicity;
source_object.coarse_ap = ap_debug;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function coarse_aperiodicity = EstimateOneSlice(x, fs, f0,...
    frequency_interval, temporal_position, fft_size, number_of_aperiodicity,...
    window)
if f0 == 0;
  coarse_aperiodicity = zeros(number_of_aperiodicity, 1);
  return;
end;

static_centroid =...
  CalculateStaticCentroid(x, fs, f0, temporal_position, fft_size);
waveform = CalculateWaveform(x, fs, f0, temporal_position, 2, 1);
smoothed_power_spectrum =...
  CalculateSmoothedPowerSpectrum(waveform, fs, f0, fft_size);
static_group_delay =...
  CalculateStaticGroupDelay(static_centroid, smoothed_power_spectrum, fs, f0,...
  fft_size);
coarse_aperiodicity =...
  CalculateCoarseAperiodicity(static_group_delay, fs, fft_size,...
  frequency_interval, number_of_aperiodicity, window);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function waveform = CalculateWaveform(x, fs, f0, temporal_position,...
    half_length, window_type) % 1: hanning, 2: blackman
%  prepare internal variables
fragment_index = 0 : round(half_length * fs / f0);
number_of_fragments = length(fragment_index);
base_index = [-fragment_index(number_of_fragments : -1 : 2), fragment_index]';
index = temporal_position * fs + 1 + base_index;
safe_index = min(length(x), max(1, round(index)));

%  wave segments and set of windows preparation
segment = x(safe_index);
time_axis = base_index / fs / half_length +...
  (temporal_position * fs -round(temporal_position * fs)) / fs;
if window_type == 1 % hanning
  window = 0.5 * cos(pi * time_axis * f0) + 0.5;
else % blackman
  window = 0.08 * cos(pi * time_axis * f0 * 2) +...
    0.5 * cos(pi * time_axis * f0) + 0.42;
end;
waveform = segment .* window - window * mean(segment .* window) / mean(window);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function spectral_envelope = CalculateSmoothedPowerSpectrum(waveform, fs,...
    f0, fft_size)
power_spectrum = abs(fft(waveform, fft_size)) .^ 2;
spectral_envelope = DCCorrection(power_spectrum, fs, fft_size, f0);
spectral_envelope = LinearSmoothing(spectral_envelope, fs, fft_size, f0);
spectral_envelope = [spectral_envelope; spectral_envelope(end - 1 : -1 : 2)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function centroid = CalculateStaticCentroid(x, fs, f0, temporal_position,...
    fft_size)
waveform1 =...
  CalculateWaveform(x, fs, f0, temporal_position + 1 / f0 / 4, 2, 2);
waveform2 =...
  CalculateWaveform(x, fs, f0, temporal_position - 1 / f0 / 4, 2, 2);
centroid1 = CalculateCentroid(waveform1, fft_size);
centroid2 = CalculateCentroid(waveform2, fft_size);
centroid = DCCorrection(centroid1 + centroid2, fs, fft_size, f0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function centroid = CalculateCentroid(x, fft_size)
time_axis = (1 : length(x))';
x = x(:) ./ sqrt(sum(x.^2));

% Centroid calculation on frequency domain.
spectrum = fft(x, fft_size);
weighted_spectrum = fft(-x .* time_axis * 1i, fft_size);
centroid = -imag(weighted_spectrum) .* real(spectrum) +...
  imag(spectrum) .* real(weighted_spectrum);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function group_delay = CalculateStaticGroupDelay(static_centroid,...
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
low_frequency_axis = frequency_axis(frequency_axis < 1.2 * f0);
low_frequency_replica = interp1(f0 - low_frequency_axis,...
  signal(frequency_axis < 1.2 * f0),...
low_frequency_axis(:), 'linear', 'extrap');

signal(frequency_axis < f0) =...
  low_frequency_replica(frequency_axis < f0) + signal(frequency_axis < f0);

signal(end : -1 : fft_size / 2 + 2) = signal(2 : fft_size / 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function coarse_aperiodicity = CalculateCoarseAperiodicity(group_delay, fs,...
    fft_size, frequency_interval, number_of_aperiodicity, window)
boundary = round(fft_size / length(window) * 8);

half_window_length = floor(length(window) / 2);
coarse_aperiodicity = zeros(number_of_aperiodicity, 1);
for i = 1 : number_of_aperiodicity
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