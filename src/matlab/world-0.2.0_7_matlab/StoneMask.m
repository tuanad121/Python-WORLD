function refined_f0 = StoneMask(x, fs, temporal_positions, f0)
% Refine F0 by instantaneous frequency
% refined_f0 = StoneMask(x, fs, temporal_positions, f0)
%
% Inputs
%   x  : input signal
%   fs : sampling frequency
%   temporal_positions : Temporal positions in each f0
%   f0 : F0 estimated from an estimator.
% Output
%   refined_f0 : Refined f0
%
% 2015/12/02: First version was released.
% 2016/01/06: A part of processes was fixed.

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
function refined_f0 = GetRefinedF0(x, fs, current_time, f0_initial)

half_window_length = ceil(3 * fs / f0_initial / 2);
window_length_in_time = (2 * half_window_length + 1) / fs;
base_time = (-half_window_length : half_window_length)' / fs;
fft_size = 2 ^ ceil(log2((half_window_length * 2 + 1)) + 1);
fx = ((0 : fft_size - 1) / fft_size * fs)';

index_raw = round((current_time + base_time) * fs);
index_time = (index_raw - 1) / fs;
window_time = index_time - current_time;
main_window = 0.42 + 0.5 * cos(2 * pi * window_time / window_length_in_time) +...
  0.08 * cos(4 * pi * window_time / window_length_in_time);
diff_window = -(diff([0; main_window]) + diff([main_window; 0])) / 2;

index = max(1, min(length(x), index_raw));
spectrum = fft(x(index) .* main_window, fft_size);
diff_spectrum = fft(x(index) .* diff_window, fft_size);
numerator_i = real(spectrum) .* imag(diff_spectrum) -...
  imag(spectrum) .* real(diff_spectrum);
power_spectrum = abs(spectrum) .^ 2;
instantaneous_frequency = fx + numerator_i ./ power_spectrum * fs / 2 / pi;

trim_index = (1 : 2)';
index_list_trim = round(f0_initial * fft_size / fs * trim_index) + 1;
fixp_list = instantaneous_frequency(index_list_trim);
amp_list = sqrt(power_spectrum(index_list_trim));
f0_initial = sum(amp_list .* fixp_list) / sum(amp_list .* trim_index);

if f0_initial < 0;
  refined_f0 = 0;
  return;
end;

trim_index = (1 : 6)';
index_list_trim = round(f0_initial * fft_size / fs * trim_index) + 1;
fixp_list = instantaneous_frequency(index_list_trim);
amp_list = sqrt(power_spectrum(index_list_trim));
refined_f0 = sum(amp_list .* fixp_list) / sum(amp_list .* trim_index);
