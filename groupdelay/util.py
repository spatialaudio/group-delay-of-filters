"""Numerical computation of group delay."""
import numpy as np
from scipy.signal import tf2sos
from scipy.signal.filter_design import _validate_sos


def group_delays(b, a, w, plot=None):
    """
    Compute group delay of analog filter.

    Parameters
    ----------
    b : array_like
        Numerator of a linear filter.
    a : array_like
        Denominator of a linear filter.
    w : array_like
        Angular frequencies in rad/s.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `gd` are passed to plot.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `gd` was computed.
    gd : ndarray
        The group delay in seconds.
    """
    b, a = map(np.atleast_1d, (b, a))
    sos = tf2sos(b, a)
    gd = sos_group_delays(sos, w)[1]
    if plot is not None:
        plot(w, gd)
    return w, gd


def sos_group_delays(sos, w, plot=None):
    """
    Compute group delay of analog filter in SOS format.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    w : array_like
        Angular frequencies in rad/s.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `gd` are passed to plot.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `gd` was computed.
    gd : ndarray
        The group delay in seconds.
    """
    sos, n_sections = _validate_sos(sos)
    if n_sections == 0:
        raise ValueError('Cannot compute group delay with no sections')
    gd = 0
    for biquad in sos:
        gd += quadfilt_group_delays(biquad[:3], w)[1]
        gd -= quadfilt_group_delays(biquad[3:], w)[1]
    if plot is not None:
        plot(w, gd)
    return w, gd


def quadfilt_group_delays(b, w):
    """
    Compute group delay of 2nd-order analog filter.

    Parameters
    ----------
    b : array_like
        Coefficients of a 2nd-order analog filter.
    w : array_like
        Angular frequencies in rad/s.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `gd` was computed.
    gd : ndarray
        The group delay in seconds.
    """
    b = np.atleast_1d(b)
    w2 = w**2
    w4 = w**4
    kw_tol = dict(rtol=1e-12, atol=1e-16)
    if b[0] == 0:
        if np.isclose(b[1], 0, **kw_tol) or np.isclose(b[2], 0, **kw_tol):
            return w, np.zeros_like(w)
        else:
            return w, -(b[1]*b[2]) / (b[1]**2*w2 + b[2]**2)
    elif np.isclose(b[2], 0, **kw_tol):
        if np.isclose(b[1], 0, **kw_tol):
            return w, np.zeros_like(w)
        else:
            return w, -(b[0]*b[1]) / (b[0]**2*w2 + b[1]**2)
    else:
        return w, -((b[0]*b[1]*w2 + b[1]*b[2])
                    / (b[0]**2*w4 + (b[1]**2 - 2*b[0]*b[2])*w2 + b[2]**2))


def zpk_group_delays(z, p, k, w, plot=None):
    """
    Compute group delay of analog filter in zpk format.

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter
    w : array_like
        Angular frequencies in rad/s.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `gd` are passed to plot.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `gd` was computed.
    gd : ndarray
        The group delay in seconds.
    """
    gd = 0
    for z_i in z:
        gd += zorp_group_delays(z_i, w)[1]
    for p_i in p:
        gd -= zorp_group_delays(p_i, w)[1]
    if plot is not None:
        plot(w, gd)
    return w, gd


def zorp_group_delays(zorp, w):
    """
    Compute group delay of analog filter with a single zero/pole.

    Parameters
    ----------
    zorp : complex
        Zero or pole of a 1st-order linear filter
    w : array_like
        Angular frequencies in rad/s.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `gd` was computed.
    gd : ndarray
        The group delay in seconds.
    """
    sigma, omega = np.real(zorp), np.imag(zorp)
    return w, (sigma) / (sigma**2 + (w - omega)**2)


# digital
def group_delayz(b, a, w, plot=None, fs=2*np.pi):
    """
    Compute the group delya of digital filter.

    Parameters
    ----------
    b : array_like
        Numerator of a linear filter.
    a : array_like
        Denominator of a linear filter.
    w : array_like
        Frequencies in the same units as `fs`.
    plot : callable
        A callable that takes two arguments. If given, the return parameters
        `w` and `gd` are passed to plot.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    w : ndarray
        The frequencies at which `gd` was computed, in the same units as `fs`.
    gd : ndarray
        The group delay in seconds.
    """
    b, a = map(np.atleast_1d, (b, a))
    sos = tf2sos(b, a)
    gd = sos_group_delayz(sos, w, plot, fs)[1]
    return w, gd


def sos_group_delayz(sos, w, plot=None, fs=2*np.pi):
    """
    Compute group delay of digital filter in SOS format.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    w : array_like
        Frequencies in the same units as `fs`.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `gd` are passed to plot.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    w : ndarray
        The frequencies at which `gd` was computed.
    gd : ndarray
        The group delay in seconds.
    """
    sos, n_sections = _validate_sos(sos)
    if n_sections == 0:
        raise ValueError('Cannot compute group delay with no sections')
    gd = 0
    for biquad in sos:
        gd += quadfilt_group_delayz(biquad[:3], w, fs)[1]
        gd -= quadfilt_group_delayz(biquad[3:], w, fs)[1]
    if plot is not None:
        plot(w, gd)
    return w, gd


def quadfilt_group_delayz(b, w, fs=2*np.pi):
    """
    Compute group delay of 2nd-order digital filter.

    Parameters
    ----------
    b : array_like
        Coefficients of a 2nd-order digital filter.
    w : array_like
        Frequencies in the same units as `fs`.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    w : ndarray
        The frequencies at which `gd` was computed.
    gd : ndarray
        The group delay in seconds.
    """
    W = 2 * np.pi * w / fs
    c1 = np.cos(W)
    c2 = np.cos(2*W)
    u0, u1, u2 = b**2  # b[0]**2, b[1]**2, b[2]**2
    v0, v1, v2 = b * np.roll(b, -1)  # b[0]*b[1], b[1]*b[2], b[2]*b[0]
    num = (u1+2*u2) + (v0+3*v1)*c1 + 2*v2*c2
    den = (u0+u1+u2) + 2*(v0+v1)*c1 + 2*v2*c2
    return w, 1 / fs * num / den


def zpk_group_delay(z, p, k, w, plot=None, fs=2*np.pi):
    """
    Compute group delay of digital filter in zpk format.

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter
    w : array_like
        Frequencies in the same units as `fs`.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `gd` are passed to plot.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    w : ndarray
        The frequencies at which `gd` was computed.
    gd : ndarray
        The group delay in seconds.
    """
    gd = 0
    for z_i in z:
        gd += zorp_group_delayz(z_i, w)[1]
    for p_i in p:
        gd -= zorp_group_delayz(p_i, w)[1]
    return w, gd


def zorp_group_delayz(zorp, w, fs=1):
    """
    Compute group delay of digital filter with a single zero/pole.

    Parameters
    ----------
    zorp : complex
        Zero or pole of a 1st-order linear filter
    w : array_like
        Frequencies in the same units as `fs`.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    w : ndarray
        The frequencies at which `gd` was computed.
    gd : ndarray
        The group delay in seconds.
    """
    W = 2 * np.pi * w / fs
    r, phi = np.abs(zorp), np.angle(zorp)
    r2 = r**2
    cos = np.cos(W - phi)
    return w, (r2 - r*cos) / (r2 + 1 - 2*r*cos)


def db(x, *, power=False):
    """Decibel."""
    with np.errstate(divide='ignore'):
        return (10 if power else 20) * np.log10(np.abs(x))


def s2ms(t):
    """Convert seconds to milliseconds."""
    return t * 1000
