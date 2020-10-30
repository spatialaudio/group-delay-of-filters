"""Utility functions."""

import numpy as np


def db(x, *, power=False):
    """Decibel."""
    with np.errstate(divide='ignore'):
        return (10 if power else 20) * np.log10(np.abs(x))


def s2ms(t):
    """Convert seconds to milliseconds."""
    return t * 1000
