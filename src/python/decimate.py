import numpy as np
from scipy import signal
def decimate(x, q, n=None, ftype='iir', axis=-1, zero_phase=None):
    """
    Downsample the signal after applying an anti-aliasing filter.

    By default, an order 8 Chebyshev type I filter is used. A 30 point FIR
    filter with Hamming window is used if `ftype` is 'fir'.

    Parameters
    ----------
    x : ndarray
        The signal to be downsampled, as an N-dimensional array.
    q : int
        The downsampling factor. For downsampling factors higher than 13, it is
        recommended to call `decimate` multiple times.
    n : int, optional
        The order of the filter (1 less than the length for 'fir'). Defaults to
        8 for 'iir' and 30 for 'fir'.
    ftype : str {'iir', 'fir'} or ``dlti`` instance, optional
        If 'iir' or 'fir', specifies the type of lowpass filter. If an instance
        of an `dlti` object, uses that object to filter before downsampling.
    axis : int, optional
        The axis along which to decimate.
    zero_phase : bool, optional
        Prevent phase shift by filtering with `filtfilt` instead of `lfilter`
        when using an IIR filter, and shifting the outputs back by the filter's
        group delay when using an FIR filter. A value of ``True`` is
        recommended, since a phase shift is generally not desired. Using
        ``None`` defaults to ``False`` for backwards compatibility. This
        default will change to ``True`` in a future release, so it is best to
        set this argument explicitly.

        .. versionadded:: 0.18.0

    Returns
    -------
    y : ndarray
        The down-sampled signal.

    See Also
    --------
    resample : Resample up or down using the FFT method.
    resample_poly : Resample using polyphase filtering and an FIR filter.

    Notes
    -----
    The ``zero_phase`` keyword was added in 0.18.0.
    The possibility to use instances of ``dlti`` as ``ftype`` was added in
    0.18.0.
    """

    if not isinstance(q, int):
        raise TypeError("q must be an integer")

    if n is not None and not isinstance(n, int):
        raise TypeError("n must be an integer")

    if ftype == 'fir':
        if n is None:
            n = 30
        system = signal.dlti(signal.firwin(n + 1, 1. / q, window='hamming'), 1.)
    elif ftype == 'iir':
        if n is None:
            n = 8
        system = signal.dlti(*signal.cheby1(n, 0.05, 0.8 / q))
    elif isinstance(ftype, dlti):
        system = ftype._as_tf()  # Avoids copying if already in TF form
        n = np.max((system.num.size, system.den.size)) - 1
    else:
        raise ValueError('invalid ftype')

    if zero_phase is None:
        warnings.warn(" Note: Decimate's zero_phase keyword argument will "
                      "default to True in a future release. Until then, "
                      "decimate defaults to one-way filtering for backwards "
                      "compatibility. Ideally, always set this argument "
                      "explicitly.", FutureWarning)
        zero_phase = False

    sl = [slice(None)] * x.ndim

    if len(system.den) == 1:  # FIR case
        if zero_phase:
            y = signal.resample_poly(x, 1, q, axis=axis, window=system.num)
        else:
            # upfirdn is generally faster than lfilter by a factor equal to the
            # downsampling factor, since it only calculates the needed outputs
            n_out = x.shape[axis] // q + bool(x.shape[axis] % q)
            y = signal.upfirdn(system.num, x, up=1, down=q, axis=axis)
            sl[axis] = slice(None, n_out, None)

    else:  # IIR case
        if zero_phase:
            y = signal.filtfilt(system.num, system.den, x, axis=axis, padlen=3 * (max(len(system.den), len(system.num)) - 1))

        else:
            y = signal.lfilter(system.num, system.den, x, axis=axis)
        # sl[axis] = slice(None, None, q)
        # make it the same as matlab
        nd = len(y)
        n_out = np.ceil(nd / q)
        n_beg = q - (q * n_out - nd)
        # sl[axis] = slice(None, None, q)
    return y[n_beg - 1::q]