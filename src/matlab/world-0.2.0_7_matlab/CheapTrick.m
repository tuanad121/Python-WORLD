function spectrum_paramter = CheapTrick(x, fs, source_object, option)
% Spectral envelope extraction based on an algorithm, CheapTrick.
% spectrum_paramter = CheapTrick(x, fs, source_object);
%
% Inputs
%   x  : input signal
%   fs : sampling frequency
%   source_object : source information object
% Output
%   spectrum_paramter : spectum infromation
%
% Caution: WORLD is not corresponding with TANDEM-STRAIGHT.
%          However, the difference is only the name of several parameters.
%
% 2014/04/29: First version was released.
% 2015/09/22: A parameter (q1) is controllable.

% set default parameters
f0_low_limit = 71;
default_f0 = 500;
fft_size = 2 ^ ceil(log2(3 * fs / f0_low_limit + 1));
q1 = -0.09;
if nargin == 4
  if isfield(option, 'q1') == 1;
    q1 = option.q1;
  end;
end;

temporal_positions = source_object.temporal_positions;
f0_sequence = source_object.f0;
if isfield(source_object, 'vuv')
  f0_sequence(source_object.vuv == 0) = default_f0;
end;

spectrogram = zeros(fft_size / 2 + 1, length(f0_sequence));
for i = 1:length(f0_sequence);
 spectrogram(:,i) = EstimateOneSlice(x, fs, f0_sequence(i),...
   temporal_positions(i), fft_size, f0_low_limit, q1);
end;

% output parameters
spectrum_paramter.temporal_positions = temporal_positions;
spectrum_paramter.spectrogram = spectrogram;
spectrum_paramter.fs = fs;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function spectral_envelope  = EstimateOneSlice(x, fs, f0, temporal_position,...
    fft_size, f0_low_limit, q1)

if f0 < f0_low_limit; f0 = f0_low_limit; end; % safe guard

waveform = CalculateWaveform(x, fs, f0, temporal_position);
power_spectrum = CalculatePowerSpectrum(waveform, fs, fft_size, f0);
smoothed_spectrum = LinearSmoothing(power_spectrum, f0, fs, fft_size);
spectral_envelope = SmoothingWithRecovery(...
  [smoothed_spectrum; smoothed_spectrum(end - 1 : -1 : 2)], f0, fs,...
  fft_size, q1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function power_spectrum = CalculatePowerSpectrum(waveform, fs, fft_size, f0)

power_spectrum = abs(fft(waveform(:), fft_size)) .^ 2;

% DC correction
frequency_axis = (0 : fft_size - 1)' / fft_size * fs;
low_frequency_axis = frequency_axis(frequency_axis < 1.2 * f0);
low_frequency_replica = interp1(f0 - low_frequency_axis,...
  power_spectrum(frequency_axis < 1.2 * f0),...
  low_frequency_axis(:), 'linear', 'extrap');
power_spectrum(frequency_axis < f0) =...
  low_frequency_replica(frequency_axis < f0) +...
  power_spectrum(frequency_axis < f0);
power_spectrum(end : -1 : fft_size / 2 + 2) = power_spectrum(2 : fft_size / 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function waveform = CalculateWaveform(x, fs, f0, temporal_position)

%  prepare internal variables
fragment_index = 0 : round(1.5 * fs / f0);
number_of_fragments = length(fragment_index);
base_index = [-fragment_index(number_of_fragments : -1 : 2), fragment_index]';
index = temporal_position * fs + 1 + base_index;
safe_index = min(length(x), max(1, round(index)));

%  wave segments and set of windows preparation
segment = x(safe_index);
time_axis = base_index / fs / 1.5 +...
  (temporal_position * fs -round(temporal_position * fs)) / fs;
window = 0.5 * cos(pi * time_axis * f0) + 0.5;
window = window / sqrt(sum(window .^ 2));
waveform = segment .* window - window * mean(segment .* window) / mean(window);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function smoothed_spectrum = LinearSmoothing(power_spectrum, f0, fs, fft_size)

double_frequency_axis = (0 : 2 * fft_size - 1)' / fft_size * fs - fs;
double_spectrum = [power_spectrum; power_spectrum];

double_segment = cumsum(double_spectrum * (fs / fft_size));
center_frequency = (0 : fft_size / 2)' / fft_size * fs;
low_levels = interp1H(double_frequency_axis + fs / fft_size / 2,...
  double_segment, center_frequency - f0 / 3);
high_levels = interp1H(double_frequency_axis + fs / fft_size / 2,...
  double_segment, center_frequency + f0 / 3);

smoothed_spectrum = (high_levels - low_levels) * 1.5 / f0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the imprementation of a matlab function
function yi = interp1H(x, y, xi)
delta_x = x(2) - x(1);
xi = max(x(1), min(x(end), xi));
xi_base = floor((xi - x(1)) / delta_x);
xi_fraction = (xi - x(1)) / delta_x - xi_base;
delta_y = [diff(y); 0];
yi = y(xi_base + 1) + delta_y(xi_base + 1) .* xi_fraction;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function spectral_envelope =...
    SmoothingWithRecovery(smoothed_spectrum, f0, fs, fft_size, q1)
% q1 is controllable in version 0.2.0_4.
% q1 = -0.09; % Optimized by H. Akagiri (2011)

quefrency_axis = (0 : fft_size - 1)' / fs;
smoothing_lifter = sin(pi * f0 * quefrency_axis) ./ (pi * f0 * quefrency_axis);
smoothing_lifter(fft_size / 2 + 2 : end) =...
  smoothing_lifter(fft_size / 2 : -1 : 2);
smoothing_lifter(1) = 1;

compensation_lifter =...
  (1 - 2 * q1) + 2 * q1 * cos(2 * pi * quefrency_axis * f0);
compensation_lifter(fft_size / 2 + 2 : end) =...
  compensation_lifter(fft_size / 2 : -1 : 2);
tandem_cepstrum = fft(log(smoothed_spectrum));
tmp_spectral_envelope =...
  exp(real(ifft(tandem_cepstrum .* smoothing_lifter .* compensation_lifter)));
spectral_envelope = tmp_spectral_envelope(1 : fft_size / 2 + 1);

