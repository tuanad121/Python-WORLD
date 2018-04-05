# PYTHON WORLD VOCODER: 
*************************************

This is a line-by-line implementation of WORLD vocoder (Matlab, C++) in python. It supports *python 3.0* and later.

For technical detail, please check the [website](http://ml.cs.yamanashi.ac.jp/world/english/).

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

Or open the project in [PyCharm](https://www.jetbrains.com/pycharm/) and double-click the ```requirements.txt``` in PyCharm. It will ask to install the missing libraries by itself. 

# EXAMPLE
**************

In ```example/prodosy.py```, there is an example of analysis/modification/synthesis with WORLD vocoder. 
It has some examples of pitch, duration, spectrum modification.
In ```test/speed.py```, we estimate the time of analysis.

# NOTE:
**********

* The vocoder use pitch-synchronous analysis, the size of each window is determined by fundamental frequency ```F0```. The centers of the windows are equally spaced with the distance of ```frame_period``` ms.

* The Fourier transform size (```fft_size```) is determined automatically using sampling frequency and the lowest value of F0 ```f0_floor```. 
When you want to specify your own ```fft_size```, you have to use ```f0_floor = 3.0 * fs / fft_size```. 
If you decrease ```fft_size```, the ```f0_floor``` increases. But, a high ```f0_floor``` might be not good for the analysis of male voices.

* The F0 analysis ```Harvest``` is the slowest one. It's speeded up using ```numba``` and ```python multiprocessing```. The more cores you have, the faster it can become. However, you can use your own F0 analysis. In our case, we support 3 F0 analysis: ```DIO, HARVEST, and SWIPE'```


# CONTACT US
******************


Post your questions, suggestions, and discussions to GitHub Issues.
