%%  Test script for WORLD analysis/synthesis
% 2014/04/29: First version (0.1.4) for release.
% 2015/12/02: StoneMask.m was added to the project (0.2.0_5).
% 2016/12/28: Refactoring.
% 2017/01/02: Another example was added.

[x, fs] = audioread('vaiueo2d.wav');

if 0 % You can use Dio
  f0_parameter = Dio(x, fs);
  % StoneMask is an option for improving the F0 estimation performance.
  % You can skip this processing.
  f0_parameter.f0 = StoneMask(x, fs,...
    f0_parameter.temporal_positions, f0_parameter.f0);
end;
f0_parameter = Harvest(x, fs);

spectrum_parameter = CheapTrick(x, fs, f0_parameter);
source_parameter = D4C(x, fs, f0_parameter);

y = Synthesis(source_parameter, spectrum_parameter);

return;

%% Another example (we want to modify the parameters)
[x, fs] = audioread('vaiueo2d.wav');
option_harvest.f0_floor = 40;
f0_parameter = Harvest(x, fs, option_harvest);

% If you modified the fft_size, you must also modify the option in D4C.
% The lowest F0 that WORLD can work as expected is determined by the following:
% 3.0 * fs / fft_size
option_cheaptrick.fft_size = 4096;
option_d4c.fft_size = option_cheaptrick.fft_size;
spectrum_parameter = CheapTrick(x, fs, f0_parameter, option_cheaptrick);
source_parameter = D4C(x, fs, f0_parameter, option_d4c);

y = Synthesis(source_parameter, spectrum_parameter);
