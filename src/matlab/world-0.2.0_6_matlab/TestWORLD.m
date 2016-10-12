%%  Test script for WORLD analysis/synthesis
% 2014/04/29 First version (0.1.4) for release.
% 2015/12/02 StoneMask.m was added to the project (0.2.0_5)

[x, fs] = audioread('vaiueo2d.wav');

f0_parameter = Dio(x, fs);
% StoneMask is an option for improving the F0 estimation performance.
% You can skip this processing.
f0_parameter.f0 = StoneMask(x, fs,...
  f0_parameter.temporal_positions, f0_parameter.f0);
spectrum_parameter = CheapTrick(x, fs, f0_parameter);
source_parameter = D4C(x, fs, f0_parameter);

y = Synthesis(source_parameter, spectrum_parameter);

return;

soundsc(x, fs);
pause(length(x) / fs);
soundsc(y, fs);
