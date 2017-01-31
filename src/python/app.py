# built-in imports
import sys

# 3rd-party imports
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write



# local imports
#for now, let's stay pysig independent, since we may want to distribute our code freely later
#sys.path.append('/Users/tuandinh/bakup/pysig/src/python')
#from pysig import track
sys.path.append('world')


#import pyximport; pyximport.install()

from dio import dio
from stonemask import stonemask
from cheaptrick import cheaptrick
from d4c import d4c
from havest import havest
from synthesis import synthesis
from main import World
#import pyworld as pw # official

if __name__ == '__main__':   
    name='test-mwm'
    fs, x_int16 = wavread('{}.wav'.format(name))
    x = x_int16 / (2 ** 15 - 1)
    vocoder = World()
    #time, value = vocoder.get_f0(x, fs)
    # analysis
    dat = vocoder.encode(x, fs)
    if 0:
        # global pitch scaling
        dat = vocoder.scale_pitch(dat, 2)
    if 0:
        # global duration scaling
        dat = vocoder.scale_duration(dat, 2)
    if 0:
        dat = vocoder.warp_spectrum(dat, 0.8)
    # synthesis
    dat = vocoder.decode(dat)
    import simpleaudio as sa
    snd = sa.play_buffer((dat['out'] * 2 ** 15).astype(np.int16), 1, 2, fs)
    if 1:
        # draw smt
        vocoder.draw(x, dat)
    #snd.wait_done()
    

#if 0: # profiling
    #import cProfile
    #p = cProfile.run('sp2, f0_data, y = process()', 'profile')
    #import pstats
    #p = pstats.Stats('profile')
    #p.sort_stats('cumulative').print_stats(10)
#else:
    #sp2, f0_data, source_object, y = process()

#if 0:  # pyqtgraph
    #from pyqtgraph.Qt import QtGui, QtCore
    #import pyqtgraph as pg

    #app = QtGui.QApplication([])
    #window = pg.GraphicsWindow(title="Waveform example")
    #pg.setConfigOptions(antialias=True)

    ## TODO: link
    #h = pg.PlotItem(x=t, y=x) # TODO: plot y also
    #h.setLabels(bottom='time (s)')

    #window.addItem(h, row=0, col=0)

    ## from pyqtgraph import QtCore
    #H = 20 * np.log10(sp2)  # log-magnitude
    #imi = pg.ImageItem()
    #imi.setImage(-H.T, autoLevels=True)
    ##imi.setRect(QtCore.QRect(0, 0, t[-1], fs/2))  # I believe this is the way...
    #imi.scale(t[-1] / fs / imi.width(), fs / 2 / imi.height()) # ...not this
    #imv = pg.ViewBox()
    #imv.addItem(imi)
    #imp = pg.PlotItem(viewBox=imv)
    #imp.setLabels(bottom='time (s)', left='frquency (Hz)')

    ##imp.setXLink(h) # maybe?

    #window.addItem(imp, row=1, col=0)

    #H = source_object['aperiodicity']  # log-magnitude
    #imi = pg.ImageItem()
    #imi.setImage(-H.T, autoLevels=True)
    ##imi.setRect(QtCore.QRect(0, 0, t[-1], fs/2))  # I believe this is the way...
    #imi.scale(t[-1] / fs / imi.width(), fs / 2 / imi.height()) # ...not this
    #imv = pg.ViewBox()
    #imv.addItem(imi)
    #imp = pg.PlotItem(viewBox=imv)

    #window.addItem(imp, row=2, col=0)
    #window.raise_()

    #app.exec_()

