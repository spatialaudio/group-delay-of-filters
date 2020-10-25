"""Compare frequency response and group delay of analog IIR LPFs."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, cheby2, ellip, freqz
from util import group_delayz, db, s2ms


def plot_frequency_response(f, H, gd, name, flim=None, mlim=None, tlim=None):
    """Plot magnitude and group delay response."""
    lw = 2
    fig, ax = plt.subplots(figsize=(6, 2.5), ncols=2, sharex=True,
                           gridspec_kw=dict(wspace=0.35))
    ax[0].plot(f, np.abs(H), lw=lw)
    ax[1].plot(f, s2ms(gd), lw=lw)
    for axi in ax:
        axi.grid(True)
        axi.set_xlim(flim)
        axi.set_xscale('log')
        axi.set_xticks(10**np.arange(1, 5))
        axi.set_xlabel('Freuency in Hz')
        if flim is not None:
            axi.set_xlim(flim)
    if mlim is not None:
        ax[0].set_ylim(mlim)
    if tlim is not None:
        ax[1].set_ylim(tlim)
    ax[0].set_ylabel('Magnitude')
    ax[1].set_ylabel('Group delay in ms')
    plt.savefig('LPF-{}.png'.format(name), bbox_inches='tight')
    return fig, ax


fs = 48000

# filter design
N = 4
f_cutoff = 1000
rp = 0.5  # pass band ripples (for cheby1 and ellip)
rs = 20  # stopband attenuation (for cheby2 and ellip)
kw_filter = dict(N=N, Wn=f_cutoff, btype='low', output='ba', fs=fs)
tf_butter = butter(**kw_filter)
tf_cheby1 = cheby1(rp=rp, **kw_filter)
tf_cheby2 = cheby2(rs=rs, **kw_filter)
tf_ellip = ellip(rp=rp, rs=rs, **kw_filter)

# magnitude responses
fmin, fmax, num_f = 20, 22000, 2000
f = np.logspace(np.log10(fmin), np.log10(fmax), num=num_f, endpoint=True)
kw_freqz = dict(worN=f, fs=fs)
_, H_butter = freqz(*tf_butter, **kw_freqz)
_, H_cheby1 = freqz(*tf_cheby1, **kw_freqz)
_, H_cheby2 = freqz(*tf_cheby2, **kw_freqz)
_, H_ellip = freqz(*tf_ellip, **kw_freqz)

# group delay
kw_gd = dict(w=f, fs=fs)
_, gd_butter = group_delayz(*tf_butter, **kw_gd)
_, gd_cheby1 = group_delayz(*tf_cheby1, **kw_gd)
_, gd_cheby2 = group_delayz(*tf_cheby2, **kw_gd)
_, gd_ellip = group_delayz(*tf_ellip, **kw_gd)

# plots
flim = fmin, fmax
mlim = -0.1, 1.1
taulim = -0.2, 3.6
tlim = -0.1, 8.1
voffset = 0.12
names = ('Butterworth', 'Chebyshev-I', 'Chebyshev-II', 'Elliptic')
H = (H_butter, H_cheby1, H_cheby2, H_ellip)
gd = (gd_butter, gd_cheby1, gd_cheby2, gd_ellip)

for (Hi, gdi, namei) in zip(H, gd, names):
    plot_frequency_response(f, Hi, gdi, namei, flim, mlim, taulim)
