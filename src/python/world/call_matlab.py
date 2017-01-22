from pymatbridge import Matlab # pip install pymatbridge
from numpy import genfromtxt

def call_matlab(filename):
    mlab = Matlab(executable='/Applications/MATLAB_R2016b.app/bin/matlab')
    mlab.start()
    matlab_code = \
'''
addpath('../../matlab/world-0.2.1_matlab');
[x, fs] = audioread('{}');
if 0 % You can use Dio
f0_parameter = Dio(x, fs);
% StoneMask is an option for improving the F0 estimation performance.
% You can skip this processing.
f0_parameter.f0 = StoneMask(x, fs,...
  f0_parameter.temporal_positions, f0_parameter.f0);
else
f0_parameter = Harvest(x, fs);
end;
dlmwrite('f0.csv',f0_parameter.f0);

spectrum_parameter = CheapTrick(x, fs, f0_parameter);
dlmwrite('spec.csv', spectrum_parameter.spectrogram);
%source_parameter = D4C(x, fs, f0_parameter);

%y = Synthesis(source_parameter, spectrum_parameter);

'''.format(filename)
    mlab.run_code(matlab_code)
    mlab.stop()
    
    f0 = genfromtxt('f0.csv', delimiter=',') 
    spec = genfromtxt('spec.csv', delimiter=',')
    from os import remove
    remove('f0.csv')
    remove('spec.csv')
    return f0, spec

if __name__ == '__main__':
    f0, spec = call_matlab('test/test-mwm.wav')