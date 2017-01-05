#from dio import dio
from havest import havest
from scipy.io.wavfile import read as wavread

path = 'test/test-mwm.wav'
fs, x_int16 = wavread(path)
x = x_int16 / (2 ** 15 - 1)
f0_data = havest(x, fs)
print('haha')