import numpy as np
from scipy.io.wavfile import read

from decimal import Decimal, ROUND_HALF_UP
from scipy import interpolate
from matplotlib import mlab


def swipe(fs, x, plim=[71, 800], dt=0.005, sTHR=float('-inf')):
    plim = np.array(plim)
    dlog2p = 1/96
    dERBs = 0.1
    num_samples = int(1000 * len(x) / fs / (dt * 1000) + 1)
    t = np.arange(0, num_samples) * dt
    dc = 4
    K = 2 # parameter k for Hann window
    # Define pitch candidates
    log2pc = np.arange(np.log2(plim[0]) * 96, np.log2(plim[-1]) * 96)
    log2pc *= dlog2p
    pc = 2 ** log2pc
    S = np.zeros((len(pc), len(t))) # Pitch strength matrix
    # Determine P2-WSs
    logWs = [round_matlab(elm) for elm in np.log2(4*K*fs/plim)]
    ws = 2 ** np.arange(logWs[0],logWs[1]-1,-1) # P2-WSs
    p0 = 4 *K * fs / ws # Optimal pitches for P2-WSs
    # Determine window size used by each pitch candidate
    d = 1 + log2pc - np.log2(4 *K * fs / ws[0])
    # Create ERBs spaced frequencies (in Hertz)
    fERBs = erbs2hz(np.arange(hz2erbs(pc[0] / 4), hz2erbs(fs/2), dERBs))
    for i in range(len(ws)):
        dn = round_matlab(dc * fs / p0[i]) # Hop size in samples
        # Zero pad signal
        xzp = np.r_[np.zeros(int(ws[i]/2)), np.r_[x, np.zeros(int(dn+ws[i]/2))]]
        # Compute spectrum
        w = np.hanning(ws[i]+2)[1:-1]
        o = max(0, np.round(ws[i] - dn)) # Window overlap
        X, f, ti = mlab.specgram(x=xzp, NFFT=ws[i], Fs=fs, window=w, noverlap=o, mode='complex')

        ti = np.r_[0, ti[:-1]]
        # Interplolate at equidistant ERBs steps
        M = np.maximum(0, interpolate.interp1d(f, np.abs(X.T), kind='cubic')(fERBs)) # Magnitude
        M = M.T
        L =np.sqrt(M) # Loudness
        # Select candidates that use this window size
        if i==len(ws) - 1:
            j = np.where(d - (i + 1) > -1)[0]
            k = np.where(d[j] - (i + 1) < 0)[0]
        elif i==0:
            j = np.where(d - (i + 1) < 1)[0]
            k = np.where(d[j] - (i + 1) > 0)[0]
        else:
            j = np.where(np.abs(d - (i + 1)) < 1)[0]
            k = np.arange(len(j))
        Si = pitchStrengthAllCandidates(fERBs, L, pc[j])
        ## TODO: check followings
        if Si.shape[1] > 1:
            Si = interpolate.interp1d(ti, Si, bounds_error=False, fill_value='nan')(t)
        else:
            Si = np.matlib.repmat( np.nan, len(Si), len(t) )
        _lambda = d[j[k]] - i - 1
        mu = np.ones( j.shape )
        mu[k] = 1 - np.abs( _lambda )
        S[j,:] = S[j,:] + np.matlib.repmat(mu.reshape(-1,1), 1, Si.shape[1]) * Si
    # Fine tune the pitch using parabolic interpolation
    p = np.matlib.repmat(np.nan, S.shape[1], 1)
    s = np.matlib.repmat(np.nan, S.shape[1], 1)
    for j in range(S.shape[1]):
        s[j] = np.max(S[:,j])
        i = np.argmax(S[:,j])
        if s[j] < sTHR: continue
        if i== 0:
            p[j] = pc[0]
        elif i == len(pc) - 1:
            p[j] = pc[0]
        else:
            I = np.arange(i-1, i+2)
            tc = 1 / pc[I]
            ntc = (tc / tc[1] - 1) * 2 * np.pi
            idx = np.isfinite(S[I, j])
            c = np.zeros(len(ntc))
            c += np.nan
            ntc_ = ntc[idx]
            I_ = I[idx]
            if len(I_) < 2:
                c[idx] = (S[I, j])[0] / ntc[0]
            else:
                c[idx] = np.polyfit(ntc_, (S[I_, j]), 2)
            ftc = 1 / (2 ** np.arange(np.log2(pc[I[0]]), np.log2(pc[I[2]]) + 1/12/64,1/12/64))
            nftc = (ftc / tc[1] - 1) * 2 * np.pi
            pval = np.polyval(c, nftc)
            s[j] = np.max(pval)
            k = np.argmax(pval)
            p[j] = 2 ** ( np.log2(pc[I[0]]) + (k)/12/64 )
    p = p.flatten()
    p[np.isnan(p)] = 0
    vuv = np.zeros_like(p)
    vuv[p>0] = 1
    return {
        'temporal_positions':t,
        'f0': p,
        'vuv': vuv
    }


def round_matlab(n):
    '''
    this function works as Matlab round() function
    python round function choose the nearest even number to n, which is different from Matlab round function
    :param n: input number
    :return: rounded n
    '''
    return int(Decimal(n).quantize(0, ROUND_HALF_UP))

def pitchStrengthAllCandidates(f, L, pc):
    # Normalize loudness
    if np.any(L==0):
        print('')
    den = np.sqrt(np.sum(L * L, axis=0))
    den = np.where(den==0, 2.220446049250313e-16, den)
    L = L / den
    # Create pitch salience matrix
    S = np.zeros((len(pc), L.shape[1]))
    for j in range(len(pc)):
        S[j,:] = pitchStrengthOneCandidate(f, L, pc[j])
    return S
def pitchStrengthOneCandidate(f, L, pc):
    n = int(np.fix(f[-1] / pc - 0.75))
    # number of harmonics
    k = np.zeros(len(f)) # Kernel
    q = f / pc # Normalize frequency w.r.t candidate
    for i in ([1] + sieve(n)):
        a = np.abs(q-i)
        # Peak's weig
        p = a < 0.25
        k[p] = np.cos(2 * np.pi * q[p])
        # Valleys' weights
        v = np.logical_and((0.25 < a), (a < 0.75))
        k[v] = k[v] + np.cos(2 * np.pi * q[v]) /2
    # Apply envelope
    k *= np.sqrt(1 / f)
    # K+-normalize kernel
    k /= np.linalg.norm(k[k>0])
    # Compute pitch strength
    S = k @ L
    return S
def hz2erbs(hz):
    erbs = 21.4 * np.log10(1 + hz / 229)
    return erbs
def erbs2hz(erbs):
    hz = (10 ** (erbs / 21.4) - 1) * 229
    return hz
from math import sqrt

def sieve(n):
    # returns all primes between 2 and n
    primes = list(range(2,n+1))
    max = sqrt(n)
    num = 2
    while num < max:
        i = num
        while i <= n:
            i += num
            if i in primes:
                primes.remove(i)
        for j in primes:
            if j > num:
                num = j
                break
    return primes

if __name__ == '__main__':
    from matplotlib import mlab

    fs, x = read('arctic_a0001.wav')
    x = x / (2 ** 15 - 1)
    source = swipe(fs, x, [71, 800], 0.005, 0.3)
    from matplotlib import pyplot as plt
    plt.plot(source['f0'])
    plt.show()
