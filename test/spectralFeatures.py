# built-in imports
import timeit

# 3rd-party imports
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write

# local imports
from world import main

def lsd(ori_spec, syn_spec):
    # N x D orig_spec and N x D syn_spec; N is # frames
    # remove energy
    ori_tmp = ori_spec / np.sqrt(np.mean(ori_spec ** 2, axis=1)).reshape(-1, 1)
    syn_tmp = syn_spec / np.sqrt(np.mean(syn_spec ** 2, axis=1)).reshape(-1, 1)
    # log spectral distortion
    result = np.mean((np.mean((20 * np.log10(ori_tmp) - 20 * np.log10(syn_tmp)) ** 2, axis=1)) ** 0.5)
    return result

fs, x_int16 = wavread('test-mwm.wav')
x = x_int16 / (2 ** 15 - 1)
vocoder = main.World()

data = vocoder.encode(fs, x, f0_method='harvest')
if 0: #  log filterbank
    lfbank = vocoder.encode_lfbank(data['spectrogram'].T)
    assert lfbank.shape[1] == 32

if 0: #  mcep
    mcep = vocoder.encode_mcep(data['spectrogram'].T, n0=40)
    assert mcep.shape[1] == 40
    spec = vocoder.decode_mcep(mcep, fft_size=1024)
    print(f"Log-spectral distortion {lsd(spec, data['spectrogram'].T)} dB")  # 5.23 dB

if 1: #  manifold vocoder
    from keras.models import load_model
    encoder = load_model('../manifold/timit_vae_encoder_0001')
    decoder = load_model('../manifold/timit_vae_decoder_0001')

    mcep = vocoder.encode_mcep(data['spectrogram'].T, n0=40)
    # we exclude zeroth-coefficients to calculate mean
    # in the paper, it's the mean MCEP of all database not just one file
    m = np.mean(mcep[:,1:], axis=0)  
    energy = mcep[:, 0]
    mcep = mcep[:, 1:40]
    Zc, Yc = vocoder.encode_vae(mcep, energy, encoder=encoder, decoder=decoder,
                    window=0, n0=40, batch_size=256, mean=m)  # Zc is the latent variable, Yc is decoded MCEP from Zc
    print(f'MCEP shape {mcep.shape}, latent variable shape {Zc.shape}')
    spec = vocoder.decode_mcep(Yc, fft_size=1024)
    print(f"Log-spectral distortion {lsd(spec, data['spectrogram'].T)} dB")  # 9.62 dB
