# PYTHON WORLD VOCODER: 
*************************************

This is a line-by-line implementation of WORLD vocoder (Matlab, C++) in python. It supports *python 3.0* and later.

For technical detail, please check the [website](http://www.kki.yamanashi.ac.jp/~mmorise/world/english/).

# INSTALATION
*********************

Python WORLD uses the following dependencies:

* numpy, scipy
* matplotlib
* numba
* simpleaudio (just for demonstration)

Install python dependencies:

```
pip install -r requirements.txt
```

Or import the project with [PyCharm](https://www.jetbrains.com/pycharm/) and open ```requirements.txt``` in PyCharm. 
It will ask to install the missing libraries by itself. 

# EXAMPLE
**************

The easiest way to run those examples is to import the ```Python-WORLD``` folder into PyCharm.

In ```example/prodosy.py```, there is an example of analysis/modification/synthesis with WORLD vocoder. 
It has some examples of pitch, duration, spectrum modification.

First, we read an audio file:

```python
from scipy.io.wavfile import read as wavread
fs, x_int16 = wavread(wav_path)
x = x_int16 / (2 ** 15 - 1) # to float
```

Then, we declare a vocoder and encode the audio file:

```python
from world import main
vocoder = main.World()
# analysis
dat = vocoder.encode(fs, x, f0_method='harvest')
```

in which, ```fs``` is sampling frequency and ```x``` is the speech signal.

The ```dat``` is a dictionary object that contains pitch, magnitude spectrum, and aperiodicity. 

We can scale the pitch:

```python
dat = vocoder.scale_pitch(dat, 1.5)
```

Be careful when you scale the pich because there is upper limit and lower limit.

We can make speech faster or slower:

```python
dat = vocoder.scale_duration(dat, 2)
```

In ```test/speed.py```, we estimate the time of analysis.

To use d4c_requiem analysis and requiem_synthesis in WORLD version 0.2.2, set the variable ```is_requiem=True```:

```python
# requiem analysis
dat = vocoder.encode(fs, x, f0_method='harvest', is_requiem=True)
```

To extract log-filterbanks, MCEP-40, VAE-12 as described in the paper `Using a Manifold Vocoder for Spectral Voice and Style Conversion`, check ```test/spectralFeatures.py```. You need Keras 2.2.4 and TensorFlow 1.14.0 to extract VAE-12.
Check out [speech samples](https://tuanad121.github.io/samples/2019-09-15-Manifold/)

# NOTE:
**********

* The vocoder use pitch-synchronous analysis, the size of each window is determined by fundamental frequency ```F0```. The centers of the windows are equally spaced with the distance of ```frame_period``` ms.

* The Fourier transform size (```fft_size```) is determined automatically using sampling frequency and the lowest value of F0 ```f0_floor```. 
When you want to specify your own ```fft_size```, you have to use ```f0_floor = 3.0 * fs / fft_size```. 
If you decrease ```fft_size```, the ```f0_floor``` increases. But, a high ```f0_floor``` might be not good for the analysis of male voices.

* The F0 analysis ```Harvest``` is the slowest one. It's speeded up using ```numba``` and ```python multiprocessing```. The more cores you have, the faster it can become. However, you can use your own F0 analysis. In our case, we support 3 F0 analysis: ```DIO, HARVEST, and SWIPE'```


# CITATION:

If you find the code helpful and want to cite it, please use:

Dinh, T., Kain, A., & Tjaden, K. (2019). Using a manifold vocoder for spectral voice and style conversion. Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, 2019-September, 1388-1392.


# CONTACT US
******************


Post your questions, suggestions, and discussions to GitHub Issues.
