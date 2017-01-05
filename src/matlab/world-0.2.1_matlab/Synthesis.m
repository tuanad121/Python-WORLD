function y = Synthesis(source_object, filter_object)
% Waveform synthesis from the estimated parameters
% y = Synthesis(source_object, filter_object)
%
% Input
%   source_object : F0 and aperiodicity
%   filter_object : spectral envelope
%
% Output
%   y : synthesized waveform
%
% 2016/12/28: Refactoring

vuv = source_object.vuv;
spectrogram = filter_object.spectrogram;
default_f0 = 500;
f0 = source_object.f0;
fs = filter_object.fs;
temporal_positions = source_object.temporal_positions;

time_axis = temporal_positions(1) : 1 / fs : temporal_positions(end);
y = 0 * time_axis';

[pulse_locations, pulse_locations_index, interpolated_vuv] = ...
  TimeBaseGeneration(temporal_positions, f0, fs, vuv, time_axis, default_f0);

fft_size = (size(spectrogram, 1) - 1) * 2;
base_index = -fft_size / 2 + 1 : fft_size / 2;
y_length = length(y);
tmp_complex_cepstrum = zeros(fft_size, 1);
latter_index = fft_size / 2 + 1 : fft_size;

temporal_position_index = interp1(temporal_positions, ...
  1 : length(temporal_positions), pulse_locations, 'linear', 'extrap');
temporal_position_index = max(1, min(length(temporal_positions),...
  temporal_position_index));

amplitude_aperiodic = source_object.aperiodicity .^ 2;
amplitude_periodic = max(0.001, (1 - amplitude_aperiodic));

for i = 1 : length(pulse_locations_index)
  [spectrum_slice, periodic_slice, aperiodic_slice] = ...
    GetSpectralParameters(temporal_positions, temporal_position_index(i),...
    spectrogram, amplitude_periodic, amplitude_aperiodic, pulse_locations(i));
  
  noise_size = ...
    pulse_locations_index(min(length(pulse_locations_index), i + 1)) -...
    pulse_locations_index(i);
  output_buffer_index = ...
    max(1, min(y_length, pulse_locations_index(i) + base_index));
  
  if interpolated_vuv(pulse_locations_index(i)) >= 0.5
    tmp_periodic_spectrum = spectrum_slice .* periodic_slice;
    tmp_periodic_spectrum(tmp_periodic_spectrum == 0) = eps;
    periodic_spectrum =...
      [tmp_periodic_spectrum; tmp_periodic_spectrum(end - 1 : -1 : 2)];
    
    tmp_cepstrum = real(fft(log(abs(periodic_spectrum)') / 2));
    tmp_complex_cepstrum(latter_index) = tmp_cepstrum(latter_index) * 2;
    tmp_complex_cepstrum(1) = tmp_cepstrum(1);
    
    response = fftshift(real(ifft(exp(ifft(tmp_complex_cepstrum)))));
    y(output_buffer_index) =...
      y(output_buffer_index) + response * sqrt(max(1, noise_size));
    tmp_aperiodic_spectrum = spectrum_slice .* aperiodic_slice;
  else
    tmp_aperiodic_spectrum = spectrum_slice;
  end;
  
  tmp_aperiodic_spectrum(tmp_aperiodic_spectrum == 0) = eps;
  aperiodic_spectrum =...
    [tmp_aperiodic_spectrum; tmp_aperiodic_spectrum(end - 1 : -1 : 2)];
  tmp_cepstrum = real(fft(log(abs(aperiodic_spectrum)') / 2));
  tmp_complex_cepstrum(latter_index) = tmp_cepstrum(latter_index) * 2;
  tmp_complex_cepstrum(1) = tmp_cepstrum(1);
  response = fftshift(real(ifft(exp(ifft(tmp_complex_cepstrum)))));
  noise_input = randn(max(3, noise_size), 1);
  
  y(output_buffer_index) = y(output_buffer_index) +...
    fftfilt(noise_input - mean(noise_input), response);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pulse_locations, pulse_locations_index, vuv_interpolated] = ...
  TimeBaseGeneration(temporal_positions, f0, fs, vuv, time_axis, default_f0)

f0_interpolated_raw = ...
  interp1(temporal_positions, f0, time_axis, 'linear', 'extrap');
vuv_interpolated = ...
  interp1(temporal_positions, vuv, time_axis, 'linear', 'extrap');
vuv_interpolated = vuv_interpolated > 0.5;
f0_interpolated = f0_interpolated_raw .* vuv_interpolated;
f0_interpolated(f0_interpolated == 0) = ...
  f0_interpolated(f0_interpolated == 0) + default_f0;

total_phase = cumsum(2 * pi * f0_interpolated / fs);
pulse_locations = time_axis(abs(diff(rem(total_phase, 2 * pi))) > pi / 2);
pulse_locations_index = round(pulse_locations * fs) + 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [spectrum_slice, periodic_slice, aperiodic_slice] = ...
  GetSpectralParameters(temporal_positions, temporal_position_index,...
  spectrogram, amplitude_periodic, amplitude_random, pulse_locations)
floor_index = floor(temporal_position_index);
ceil_index = ceil(temporal_position_index);
t1 = temporal_positions(floor_index);
t2 = temporal_positions(ceil_index);

if t1 == t2
  spectrum_slice = spectrogram(:, floor_index);
  periodic_slice = amplitude_periodic(:, floor_index);
  aperiodic_slice = amplitude_random(:, floor_index);
else
  spectrum_slice = ...
    interp1q([t1 t2], [spectrogram(:, floor_index) ...
    spectrogram(:, ceil_index)]', max(t1, min(t2, pulse_locations)))';
  periodic_slice = ...
    interp1q([t1 t2], [amplitude_periodic(:, floor_index) ...
    amplitude_periodic(:, ceil_index)]', max(t1, min(t2, pulse_locations)))';
  aperiodic_slice = ...
    interp1q([t1 t2], [amplitude_random(:, floor_index) ...
    amplitude_random(:, ceil_index)]', max(t1, min(t2, pulse_locations)))';
end;
