from dio import dio
from havest import havest
from stoneMask import StoneMask
from cheapTrick import CheapTrick
from d4c import  d4c
from synthesis import synthesis
from scipy.io.wavfile import read as wavread, write as wavwrite
import numpy as np
def process():
    path = 'test/test-mwm.wav'
    fs, x_int16 = wavread(path)
    x = x_int16 / (2 ** 15 - 1)

    f0_data = havest(x, fs)
    filter_obj = CheapTrick(x, fs, f0_data)
    source_obj = d4c(x, fs, f0_data)
    y = synthesis(source_obj, filter_obj)
    if 0:
        import simpleaudio as sa
        import numpy as np
        snd = sa.play_buffer((y * 2**15).astype(np.int16), 1, 2, fs)
        snd.wait_done()
if __name__=='__main__':
    import cProfile

    p = cProfile.run('process()', 'profile')
    import pstats

    p = pstats.Stats('profile')
    p.sort_stats('cumulative').print_stats(10)
