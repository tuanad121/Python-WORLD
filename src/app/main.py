# built-in imports
import sys

# 3rd-party imports
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write

# local imports
#for now, let's stay pysig independent, since we may want to distribute our code freely later
#sys.path.append('/Users/tuandinh/bakup/pysig/src/python')
sys.path.append('../python')

#from pysig import track
from dio import Dio
from dio import ZeroCrossingEngine # yours
from dio import GetF0Candidates
from StoneMask import StoneMask
from cheapTrick import CheapTrick
from D4C import D4C
from Synthesis import Synthesis
import pyworld as pw # official


name='../python/test/test-mwm'
fs, x_int16 = wavread('{}.wav'.format(name))
if 1: # limit for debug
    x_int16 = x_int16[:fs]
x = x_int16 / (2 ** 15 - 1) # go to FP in range [-1, 1]

assert x.dtype == np.float

f0_data = Dio(x,fs)
#no_stonemask = np.copy(f0_data['f0'])
f0_data['f0'] = StoneMask(x, fs,f0_data['temporal_positions'], f0_data['f0'])

filter_object = CheapTrick(x, fs, f0_data)
source_object = D4C(x, fs, f0_data)
sp2 = filter_object['spectrogram']
ap2 = source_object['aperiodicity']

y = Synthesis(source_object, filter_object)

# equalize lengths for comparison
l = min(len(x), len(y))
x = x[:l]
y = y[:l]
t = np.arange(len(x)) / fs

if 0: # matplotlib
    from matplotlib import pyplot

    fig, ax = pyplot.subplots(nrows=5, figsize=(12,8), sharex=True)
    ax[0].plot(t, x)
    ax[0].plot(t, y)
    ax[0].axis(ymin=-1, ymax=1)

    ax[1].imshow(20 * np.log10(sp2), cmap=pyplot.cm.gray_r, origin='lower', extent=[0, len(x)/fs, 0, fs/2], aspect='auto')
    ax[1].set_ylabel('frequency (Hz)')

    ax[2].specgram(x, Fs=fs)

    ax[3].specgram(y, Fs=fs)

    ax[4].plot(f0_data['temporal_positions'], f0_data['f0'])
    ax[4].axis(xmin=0, xmax=t[-1])
    ax[4].set_xlabel('time (s)')
    pyplot.show()
else:
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg

    app = QtGui.QApplication([])
    window = pg.GraphicsWindow(title="Waveform example")
    pg.setConfigOptions(antialias=True)

    # TODO: link
    h = pg.PlotItem(x=t, y=x) # TODO: plot y also
    h.setLabels(bottom='time (s)')

    window.addItem(h, row=0, col=0)

    # from pyqtgraph import QtCore
    H = 20 * np.log10(sp2)  # log-magnitude
    imi = pg.ImageItem()
    imi.setImage(-H.T, autoLevels=True)
    #imi.setRect(QtCore.QRect(0, 0, t[-1], fs/2))  # I believe this is the way...
    imi.scale(t[-1] / fs / imi.width(), fs / 2 / imi.height()) # ...not this
    imv = pg.ViewBox()
    imv.addItem(imi)
    imp = pg.PlotItem(viewBox=imv)
    imp.setLabels(bottom='time (s)', left='frquency (Hz)')

    #imp.setXLink(h) # maybe?

    window.addItem(imp, row=1, col=0)
    window.raise_()

    app.exec_()

