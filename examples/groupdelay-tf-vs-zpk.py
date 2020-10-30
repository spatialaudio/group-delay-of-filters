"""Verify the group_delays (analog) functions.
- group_delays
- zpk_group_delays
- zorp_group_delays
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqs, freqs_zpk, zpk2tf
from groupdelay import group_delays, zpk_group_delays, zorp_group_delays
from groupdelay.util import s2ms, db


def higher_order_shelving_holters(Gd, N, wc=1, normalize=True):
    """Higher-order shelving filter design by Martin Holters."""
    g = 10**(Gd / 20)
    alpha = np.stack([np.pi * (0.5 - (2*m+1)/2/N) for m in range(N)])
    p = -np.exp(1j * alpha)
    z = g**(1 / N) * p
    k = 1
    if normalize:
        z *= g**(-0.5 / N)
        p *= g**(-0.5 / N)
    return z * wc, p * wc, k


# shelving filter
N = 4
f_cutoff = 1000
w_cutoff = 2 * np.pi * f_cutoff
Gd = -12
z, p, k = higher_order_shelving_holters(Gd, N, wc=w_cutoff, normalize=True)
b, a = zpk2tf(z, p, k)

# frequency response
fmin, fmax, num_f = 20, 22000, 2000
f = np.logspace(np.log10(fmin), np.log10(fmax), num=num_f, endpoint=True)
w = 2 * np.pi * f
_, H_zpk = freqs_zpk(z, p, k, worN=w)
_, gd_zpk = zpk_group_delays(z, p, k, w=w)
_, H_tf = freqs(b, a, worN=w)
_, gd_tf = group_delays(b, a, w=w)
gd_zorp = 0  # adding up group delays of individual zeros and poles
for z_i in z:
    gd_zorp += zorp_group_delays(z_i, w=w)[1]
for p_i in p:
    gd_zorp -= zorp_group_delays(p_i, w=w)[1]


# plots
kw_zpk = dict(color='lightgray', lw=5, ls='-', alpha=1, label='zpk')
kw_tf = dict(color='C0', lw=2, ls='-', alpha=0.85, label='tf')
kw_zorp = dict(color='k', lw=2, ls=':', alpha=0.85, label='zorp')
flim = fmin, fmax
fig, ax = plt.subplots(figsize=(12, 3), ncols=3, sharex=False,
                       gridspec_kw=dict(wspace=0.4))

ax[0].plot(np.real(z / w_cutoff), np.imag(z / w_cutoff), 'C0o')
ax[0].plot(np.real(p / w_cutoff), np.imag(p / w_cutoff), 'C3x')
ax[0].grid(True)
ax[0].axis('equal')
ax[0].set_xlabel(r'$\Re(s / \omega_\mathrm{c})$')
ax[0].set_ylabel(r'$\Im(s / \omega_\mathrm{c})$')

ax[1].plot(f, db(H_zpk), **kw_zpk)
ax[1].plot(f, db(H_tf), **kw_tf)
ax[1].set_ylabel('Magnitude in dB')

ax[2].plot(f, s2ms(gd_zpk), **kw_zpk)
ax[2].plot(f, s2ms(gd_tf), **kw_tf)
ax[2].plot(f, s2ms(gd_zorp), **kw_zorp)
ax[2].set_ylabel('Group delay in ms')

ax[2].legend(loc='upper left')
for axi in ax[1:]:
    axi.set_xlim(flim)
    axi.set_xscale('log')
    axi.set_xlabel('Frequency in Hz')
    axi.grid(True)
plt.savefig('group-delay-tf-zpk-zorp.png', bbox_inches='tight')
