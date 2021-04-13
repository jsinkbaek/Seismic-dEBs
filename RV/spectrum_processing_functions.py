"""
This collection of functions is used for general work on wavelength reduced échelle spectra.
"""
import numpy as np
import alphashape
import localreg


def moving_median_filter(flux, window=51):
    """
    Filters the data using a moving median filter. Useful to capture continuum trends.
    :param flux:        np.ndarray size (n, ) of y-values (fluxes) for each wavelength
    :param window:      integer, size of moving median filter
    :return:            filtered_flux, np.ndarray size (n, )
    """
    from scipy.ndimage import median_filter
    return median_filter(flux, size=window, mode='reflect')


def AFS_algorithm():
    """
    https://arxiv.org/pdf/1904.10065v1.pdf
    Modeling the Echelle Spectra Continuum with Alpha Shapes and Local Regression Fitting

    Create alpha shape of spectrum. Then make local regression fit to upper boundary of alpha shape to estimate blaze.
    Divide spectrum with it. Then something more
    """
