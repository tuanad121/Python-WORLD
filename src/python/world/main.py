import logging
import sys

# 3rd party imports
import numpy as np
from scipy.io.wavfile import read as wavread
from matplotlib import pyplot as plt

# local imports
from .dio import dio
from .stonemask import stonemask
from .harvest import harvest
from .cheaptrick import cheaptrick
from .d4c import d4c
from .synthesis import synthesis
from .swipe import swipe



class World(object):
    def get_f0(self, fs: int, x: np.ndarray, f0_method: str = 'harvest', f0_floor: int = 71, f0_ceil: int = 800,
               channels_in_octave: int = 2, target_fs: int = 4000, frame_period: int = 5) -> tuple:
        """
        :param fs: sample frequency
        :param x: signal
        :param f0_method: f0 extraction method: dio, harvest
        :param f0_floor: smallest f0
        :param f0_ceil: largest f0
        :param channels_in_octave:
        :param target_fs: downsampled frequency for f0 extraction
        :param frame_period: in ms
        :return:
        """
        if f0_method == 'dio':
            source = dio(x, fs, f0_floor, f0_ceil, channels_in_octave, target_fs, frame_period)
            source['f0'] = stonemask(x, fs, source['temporal_positions'], source['f0'])
        elif f0_method == 'harvest':
            source = harvest(x, fs, f0_floor, f0_ceil, frame_period)
        elif f0_method == 'swipe':
            source = swipe(fs, x, plim=[f0_floor, f0_ceil],sTHR=0.3)
        else:
            raise Exception
        return source['temporal_positions'], source['f0'], source['vuv']  # or a dict

    @staticmethod
    def get_spectrum(fs: int, x: np.ndarray, f0_method: str = 'harvest', f0_floor: int = 71, f0_ceil: int = 800,
                     channels_in_octave: int = 2, target_fs: int = 4000, frame_period: int = 5, fft_size=None) -> dict:
        '''
        This function extract pitch-synchronous WORLD spectrogram
        :param fs: sampling frequency
        :param x: signal (in float)
        :param f0_method: dio, harvest, swipe
        :param f0_floor: f0 min
        :param f0_ceil: f0 max
        :param frame_period: frame shift
        :param fft_size: fourier transform length
        :param: channels_in_octave: channels per octave
        :return:
        '''
        if f0_method == 'dio':
            source = dio(x, fs, f0_floor, f0_ceil, channels_in_octave, target_fs, frame_period)
            source['f0'] = stonemask(x, fs, source['temporal_positions'], source['f0'])
        elif f0_method == 'harvest':
            source = harvest(x, fs, f0_floor, f0_ceil, frame_period)
        elif f0_method == 'swipe':
            source = swipe(fs, x, plim=[f0_floor, f0_ceil],sTHR=0.3)
        else:
            raise Exception
        filter = cheaptrick(x, fs, source, fft_size=fft_size)
        return {'f0': source['f0'],
            'temporal_positions': source['temporal_positions'],
                'fs': fs,
                'ps spectrogram': filter['ps spectrogram'],
                'spectrogram': filter['spectrogram']}

    @staticmethod
    def encode_w_gvn_f0(fs: int, x: np.ndarray, source: dict, fft_size=None) -> dict:
        '''
        Only use this function with default fft_size because f0_floor is defied by the value
        This function extract WORLD spectrogram and aperiodicity with given F0 contour
        :param fs: sampling rate
        :param x: signal
        :param source: a dictionary contains time, f0 contour and voice/unvoice
        :param fft_size: length of Fourier transform
        :return: a dictionary contains WORLD's components
        '''
        filter = cheaptrick(x, fs, source, fft_size=fft_size)
        source = d4c(x, fs, source, fft_size_for_spectrum=fft_size)
        return {'temporal_positions': source['temporal_positions'],
                'vuv': source['vuv'],
                'f0': source['f0'],
                'fs': fs,
                'spectrogram': filter['spectrogram'],
                'aperiodicity': source['aperiodicity'],
                'coarse_ap': source['coarse_ap']
                }

    def encode(self, fs: int, x: np.ndarray, f0_method: str = 'harvest', f0_floor: int = 71, f0_ceil: int = 800,
               channels_in_octave: int = 2, target_fs: int = 4000, frame_period: int = 5,
               allowed_range: float = 0.1, fft_size=None) -> dict:
        '''
        encode speech to excitation signal, f0, spectrogram

        :param fs: sample frequency
        :param x: signal
        :param f0_method: f0 extraction method: dio, harvest
        :param f0_floor: smallest f0
        :param f0_ceil: largest f0
        :param channels_in_octave: number of channels per octave
        :param target_fs: downsampled frequency for f0 extraction
        :param frame_period: in ms
        :param allowed_range:
        :param fft_size: length of Fourier transform
        :return: a dictionary contains WORLD components
        '''
        if fft_size != None:
            f0_floor = 3.0 * fs / fft_size
        if f0_method == 'dio':
            source = dio(x, fs,
                         f0_floor=f0_floor, f0_ceil=f0_ceil, channels_in_octave=channels_in_octave, target_fs=target_fs,
                         frame_period=frame_period, allowed_range=allowed_range)
            source['f0'] = stonemask(x, fs, source['temporal_positions'], source['f0'])
        elif f0_method == 'harvest':
            source = harvest(x, fs,
                             f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_period)
        elif f0_method == 'swipe':
            source = swipe(fs, x, plim=[f0_floor, f0_ceil], sTHR=0.3)
        else:
            raise Exception
        filter = cheaptrick(x, fs, source, fft_size=fft_size)
        source = d4c(x, fs, source, fft_size_for_spectrum=fft_size)

        return {'temporal_positions': source['temporal_positions'],
                'vuv': source['vuv'],
                'fs': filter['fs'],
                'f0': source['f0'],
                'aperiodicity': source['aperiodicity'],
                'ps spectrogram': filter['ps spectrogram'],
                'spectrogram': filter['spectrogram']
                }

    def scale_pitch(self, dat: dict, factor: float) -> dict:
        '''
        the function does pitch scaling
        :param dat: WORLD components (F0, spectrogram, aperiodicity)
        :param factor: scaling factor
        :return: scaled pitch.
        '''
        dat['f0'] *= factor
        return dat

    def set_pitch(self, dat: dict, time: np.ndarray, value: np.ndarray) -> dict:
        dat['f0'] = value
        dat['temporal_positions'] = time
        return dat

    def scale_duration(self, dat: dict, factor: float) -> dict:
        '''
        the function does duration scaling
        :param dat:  WORLD components (F0, spectrogram, aperiodicity)
        :param factor: scaling factor
        :return: scaled event-time to speech up or slow down the speech
        '''
        dat['temporal_positions'] *= factor
        return dat

    def warp_spectrum(self, dat: dict, factor: float) -> dict:
        dat['spectrogram'][:] = np.array([np.interp((np.arange(0, len(s)) / len(s)) ** factor,
                                                    (np.arange(0, len(s)) / len(s)),
                                                    s)
                                          for s in dat['spectrogram'].T]).T
        return dat

    def decode(self, dat: dict) -> dict:
        '''
        This function combine WORLD components (F0, spectrogram, and aperiodicity) to make sound signal
        :param dat: contains WORLD components
        :return: a dictionary contains synthesized speech and WORLD components
        '''
        y = synthesis(dat, dat)
        m = np.max(np.abs(y))
        if m > 1.0:
            logging.info('rescaling waveform')
            y /= m
        dat['out'] = y
        return dat

    def draw(self, x: np.ndarray, dat: dict):
        '''
        An example of visualize WORLD components, original signal, synthesized signal
        '''
        fs = dat['fs']
        time = dat['temporal_positions']
        y = dat['out']

        fig, ax = plt.subplots(nrows=5, figsize=(8, 6), sharex=True)
        ax[0].set_title('input signal and resynthesized-signal')
        ax[0].plot(np.arange(len(x)) / fs, x, alpha=0.5)
        ax[0].plot(np.arange(len(y)) / fs, y, alpha=0.5)
        ax[0].set_xlabel('samples')
        ax[0].legend(['original', 'synthesis'])

        X = dat['ps spectrogram']
        X = np.where(X==0, sys.float_info.epsilon, X)
        ax[1].set_title('pitch-synchronous spectrogram')
        ax[1].imshow(20 * np.log10(np.abs(X[:X.shape[0] // 2, :])), cmap=plt.cm.gray_r, origin='lower',
                     extent=[0, len(x) / fs, 0, fs / 2], aspect='auto')
        ax[1].set_ylabel('frequency (Hz)')

        ax[2].set_title('phase spectrogram')
        ax[2].imshow(np.diff(np.unwrap(np.angle(X[:X.shape[0] // 2, :]), axis=1), axis=1), cmap=plt.cm.gray_r,
                     origin='lower',
                     extent=[0, len(x) / fs, 0, fs / 2], aspect='auto')
        ax[2].set_ylabel('frequency (Hz)')

        ax[3].set_title('WORLD spectrogram')
        Y = dat['spectrogram']
        Y = np.where(Y == 0, sys.float_info.epsilon, Y)
        ax[3].imshow(20 * np.log10(Y), cmap=plt.cm.gray_r, origin='lower',
                     extent=[0, len(x) / fs, 0, fs / 2], aspect='auto')
        ax[3].set_ylabel('frequency (Hz)')

        ax[4].set_title('WORLD fundamental frequency')
        ax[4].plot(time, dat['f0'])
        ax[4].set_ylabel('time (s)')

        plt.show()
