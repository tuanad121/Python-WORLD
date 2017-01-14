function spectrum_paramter = CheapTrick(x, fs, source_object, option)
% Spectral envelope extraction based on an algorithm, CheapTrick.
% spectrum_paramter = CheapTrick(x, fs, source_object, option);
% spectrum_paramter = CheapTrick(x, fs, source_object);
%
% Input
%   x  : input signal
%   fs : sampling frequency
%   source_object : source information object
%   option    : It has two parameters (q1 and fft_size)
%               Parameter q1 is used for the spectral recovery.
%               The lowest F0 that WORLD can work as expected is determined
%               by the following: 3.0 * fs / fft_size.
%
% Output
%   spectrum_paramter : spectum infromation
%
% Caution: WORLD is not corresponding with TANDEM-STRAIGHT.
%          However, the difference is only the name of several parameters.
%
% 2014/04/29: First version was released.
% 2015/09/22: A parameter (q1) is controllable.
% 2016/12/28: Refactoring (default value of q1 was modified. -0.09 -> -0.15)
% 2017/01/02: A parameter fft_size is controllable.

% set default parameters
f0_low_limit = 71;
default_f0 = 500;
fft_size = 2 ^ ceil(log2(3 * fs / f0_low_limit + 1));
q1 = -0.15;
if nargin == 4
  if isfield(option, 'q1') == 1
    q1 = option.q1;
  end;
  if isfield(option, 'fft_size') == 1
    fft_size = option.fft_size;
  end;
end;
f0_low_limit = fs * 3.0 / fft_size;

temporal_positions = source_object.temporal_positions;
f0_sequence = source_object.f0;
if isfield(source_object, 'vuv')
  f0_sequence(source_object.vuv == 0) = default_f0;
end;

spectrogram = zeros(fft_size / 2 + 1, length(f0_sequence));
for i = 1:length(f0_sequence)
  if f0_sequence(i) < f0_low_limit; f0_sequence(i) = default_f0; end;
  spectrogram(:,i) = EstimateOneSlice(x, fs, f0_sequence(i),...
    temporal_positions(i), fft_size, q1);
end;

% output parameters
spectrum_paramter.temporal_positions = temporal_positions;
spectrum_paramter.spectrogram = spectrogram;
spectrum_paramter.fs = fs;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function spectral_envelope  = EstimateOneSlice(x, fs, current_f0,...
  current_position, fft_size, q1)
waveform = GetWindowedWaveform(x, fs, current_f0, current_position);
power_spectrum = GetPowerSpectrum(waveform, fs, fft_size, current_f0);
smoothed_spectrum = LinearSmoothing(power_spectrum, current_f0, fs, fft_size);
spectral_envelope = SmoothingWithRecovery(...
  [smoothed_spectrum; smoothed_spectrum(end - 1 : -1 : 2)], current_f0, fs,...
  fft_size, q1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function power_spectrum = GetPowerSpectrum(waveform, fs, fft_size, f0)
power_spectrum = abs(fft(waveform(:), fft_size)) .^ 2;
% DC correction
frequency_axis = (0 : fft_size - 1)' / fft_size * fs;
low_frequency_axis = frequency_axis(frequency_axis <  f0 + fs / fft_size);
low_frequency_replica = interp1(f0 - low_frequency_axis,...
  power_spectrum(frequency_axis < f0 + fs / fft_size),...
  low_frequency_axis(:), 'linear', 'extrap');
power_spectrum(frequency_axis < f0) =...
  low_frequency_replica(frequency_axis < f0) +...
  power_spectrum(frequency_axis < f0);
power_spectrum(end : -1 : fft_size / 2 + 2) = power_spectrum(2 : fft_size / 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function waveform = GetWindowedWaveform(x, fs, current_f0, current_position)
%  prepare internal variables
half_window_length = round(1.5 * fs / current_f0);
base_index = (-half_window_length : half_window_length)';
index = round(current_position * fs + 0.001) + 1 + base_index;
safe_index = min(length(x), max(1, round(index)));

%  wave segments and set of windows preparation
segment = x(safe_index);
time_axis = base_index / fs / 1.5;
window = 0.5 * cos(pi * time_axis * current_f0) + 0.5;
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
