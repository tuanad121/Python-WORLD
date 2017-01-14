function refined_f0 = StoneMask(x, fs, temporal_positions, f0)
% Refine F0 by instantaneous frequency
% refined_f0 = StoneMask(x, fs, temporal_positions, f0)
%
% Input
%   x  : input signal
%   fs : sampling frequency
%   temporal_positions : Temporal positions in each f0
%   f0 : F0 estimated from an estimator.
%
% Output
%   refined_f0 : Refined f0
%
% 2015/12/02: First version was released.
% 2016/01/06: A part of processes was fixed.
% 2016/12/28: Refactoring

refined_f0 = f0;
for i = 1 : length(temporal_positions)
  if f0(i) ~= 0
    refined_f0(i) = GetRefinedF0(x, fs, temporal_positions(i), f0(i));
    if abs(refined_f0(i) - f0(i)) / f0(i) > 0.2
      refined_f0(i) = f0(i);
    end;
  end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function refined_f0 = GetRefinedF0(x, fs, current_position, current_f0)

initial_f0 = current_f0;
half_window_length = ceil(3 * fs / initial_f0 / 2);
window_length_in_time = (2 * half_window_length + 1) / fs;
base_time = (-half_window_length : half_window_length)' / fs;
fft_size = 2 ^ ceil(log2((half_window_length * 2 + 1)) + 1);
frequency_axis = ((0 : fft_size - 1) / fft_size * fs)';

base_index = round((current_position + base_time) * fs);
index_time = (base_index - 1) / fs;
window_time = index_time - current_position;
main_window = 0.42 + 0.5 * cos(2 * pi * window_time / window_length_in_time) +...
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

harmonics_index = (1 : 2)';
index_list = round(initial_f0 * fft_size / fs * harmonics_index) + 1;
instantaneous_frequency_list = instantaneous_frequency(index_list);
amplitude_list = sqrt(power_spectrum(index_list));
initial_f0 = sum(amplitude_list .* instantaneous_frequency_list) /...
  sum(amplitude_list .* harmonics_index);

if initial_f0 < 0
  refined_f0 = 0;
  return;
end;

harmonics_index = (1 : 6)';
index_list = round(initial_f0 * fft_size / fs * harmonics_index) + 1;
instantaneous_frequency_list = instantaneous_frequency(index_list);
amplitude_list = sqrt(power_spectrum(index_list));
refined_f0 = sum(amplitude_list .* instantaneous_frequency_list) /...
  sum(amplitude_list .* harmonics_index);
