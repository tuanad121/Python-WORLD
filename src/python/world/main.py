import numpy as np

# compare to vocoder.py


class World(object):
    def get_f0(self,  x: np.ndarray, fs: int, f0_method: str=None) -> tuple:
        # don't calc spectrum
        return t, v # or a dict

    def get_spectrum(self, x: np.ndarray, fs: int) -> dict:
        return {'time': t, 'PS spectrogram': X, 'world magnitude spectrogram': M}


    def encode(self, x: np.ndarray, fs: int) -> dict:
        return {}

    def scale_pitch(self):
        pass

    def set_pitch(self, dat: dict, time: np.ndarray, value: np.ndarray) -> dict:
        return dat

    def scale_duration(self):
        pass

    def warp_spectrum(self):
        pass

    def decode(self, dat: dict) -> dict:
        dat['out'] = y
        return dat

    def draw(self, dat: dict):
        pass