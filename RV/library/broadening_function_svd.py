"""
First edition on May 01, 2021.
@author Jeppe Sinkbæk Thomsen, Master's student in astronomy at Aarhus University.
Supervisor: Assistant Professor Karsten Frank Brogaard.

Purpose of this file is broadening function calculation using a Singular Value Decomposition of a template spectrum.
Code is primarily adapted from http://www.astro.utoronto.ca/~rucinski/SVDcookbook.html and
functions in the shazam.py library for the SONG telescope (written by Emil) TODO: find last name
The primary class to create objects from for broadening function calculation is BroadeningFunction. The rest are
convenience classes for that one.
Some of the import statements lacking below are imported from rotational_broadening_function_fitting.py.
"""

import scipy.linalg as lg
import warnings
from RV.library.rotational_broadening_function_fitting import *


class DesignMatrix:
    def __init__(self, template_spectrum, span):
        """
        Creates a Design Matrix (DesignMatrix.mat) of a template spectrum for the SVD broadening function method.
        :param template_spectrum:   np.ndarray, flux of the template spectrum that design matrix should be made on.
        :param span:                int, span or width of the broadening function. Should be odd.

        :var self.vals:     same as template_spectrum
        :var self.span:     same as span
        :var self.n:        int, size of template_spectrum
        :var self.m:        same as self.span
        :var self.mat:      np.ndarray, the created design matrix. Shape (m, n-m)
        """
        self.vals = template_spectrum
        self.span = span
        self.n = self.vals.size
        self.m = self.span
        self.mat = self.map()

    def map(self):
        """
        Map stored spectrum to a design matrix.
        :return mat: np.ndarray, the created design matrix. Shape (m, n-m)
        """
        # Matrix is shape (m, n-m)
        n, m = self.n, self.m

        if np.mod(m, 2) != 1.0:
            raise ValueError('Design Matrix span must be odd.')
        if np.mod(n, 2) != 0.0:
            raise ValueError('Number of values must be even.')

        mat = np.zeros(shape=(m, n-m+1))
        for i in range(0, m):
            mat[i, :] = self.vals[i:i+n-m+1]
        return mat.T


class SingularValueDecomposition:
    def __init__(self, template_spectrum, span):
        """
        Creates a Singular Value Decomposition of a template spectrum DesignMatrix for the SVD broadening function.

        :param template_spectrum: np.ndarray, inverted flux of the template spectrum equi-spaced in velocity space
        :param span:              int, span or width (number of elements) of the broadening function design matrix.

        :var self.design_matrix: The created DesignMatrix of the template spectrum.
        :var self.u:             np.ndarray, the unitary matrix having left singular vectors as columns.
                                 Shape (span, span)
        :var self.w:             np.ndarray, vector of the singular values, in non-increasing order.
                                 Shape (template_spectrum.size, )
        :var self.vH:            np.ndarray, the unitary matrix having right singular vectors as rows.
                                 Shape (span, template_spectrum.size)
        """
        self.design_matrix = DesignMatrix(template_spectrum, span)
        self.u, self.w, self.vH = lg.svd(self.design_matrix.mat, compute_uv=True, full_matrices=False)


class BroadeningFunction:
    def __init__(self, program_spectrum, template_spectrum, span, dv, copy=False):
        """
        Creates a broadening function object storing all the necessary variables for it.
        Using BroadeningFunction.solve(), the broadening function is found by solving the Singular Value
        Decomposition of a template spectrum.

        :param program_spectrum:  np.ndarray, inverted flux of the program spectrum equi-spaced in velocity space
        :param template_spectrum: np.ndarray, inverted flux of the template spectrum equi-spaced in velocity space
        :param span:              int, span or width (number of elements) of the broadening function design matrix.
        :param dv:                float, the velocity spectrum resolution in km/s
        :param copy:              bool, controls if initial calculations should not be done (in case the result is to be
                                  copied from another object)

        :var self.spectrum:  same as program_spectrum
        :var self.svd:       a created SingularValueDecomposition of the template spectrum.
        :var self.bf:        None (before self.solve() is run). After, np.ndarray, the calculated broadening function.
        :var self.bf_smooth: None (before self.smooth() is run). After, np.ndarray, smoothed broadening function.
        :var self.velocity:  np.ndarray, velocity values of the broadening function in km/s.
        """
        # TODO: Ask about rvr (instead of span/bn)
        if np.mod(span, 2) != 1.0:
            warnings.warn('Warning: Design Matrix span must be odd. Lengthening by 1.')
            span += 1
        if template_spectrum.size != program_spectrum.size:
            raise ValueError(f'template_spectrum.size does not match program_spectrum.size. Size '
                             f'{template_spectrum.size} vs Size {program_spectrum.size}')
        if np.mod(template_spectrum.size, 2) != 0.0:
            warnings.warn('Warning: template_spectrum length must be even. Shortening by 1.')
            template_spectrum = template_spectrum[:-1]
            program_spectrum = program_spectrum[:-1]
        self.spectrum = program_spectrum
        self.template_spectrum = template_spectrum
        self.span = span
        self.dv = dv
        if ~copy:
            self.svd = SingularValueDecomposition(template_spectrum, span)
        else:
            self.svd = None
        self.bf = None
        self.bf_smooth = None
        self.smooth_sigma = 5.0
        if ~copy:
            self.velocity = -np.arange(-int(span/2), int(span/2+1))*dv
        else:
            self.velocity = None
        self.fit = None
        self.model_values = None

    def __copy__(self):
        new_copy = type(self)(self.spectrum, self.template_spectrum, self.span, self.dv, copy=True)
        new_copy.svd = self.svd
        new_copy.bf = self.bf
        new_copy.bf_smooth = self.bf_smooth
        new_copy.smooth_sigma = self.smooth_sigma
        new_copy.velocity = self.velocity
        new_copy.fit = self.fit
        new_copy.model_values = self.model_values
        return new_copy

    @staticmethod
    def truncate(spectrum, design_matrix):
        """
        Truncates a program spectrum, which is essential before solving the linear equations.
        :param spectrum:        np.ndarray, the input spectrum
        :param design_matrix:   np.ndarray, the design matrix which is to be used
        """
        m = design_matrix.m
        return spectrum[int(m/2):-int(m/2)]

    def solve(self):
        """
        Solves the system involving the Singular Value Decomposition of the template spectrum, and the program spectrum,
        to provide the broadening function of the program spectrum.
        :return: np.ndarray, broadening function of the program spectrum
        """
        spectrum_truncated = self.truncate(self.spectrum, self.svd.design_matrix)
        u, w, vH = self.svd.u, self.svd.w, self.svd.vH

        limit_w = 0.0
        w_inverse = 1.0/w
        limit_mask = (w < limit_w)
        w_inverse[limit_mask] = 0.0
        diag_mat_w_inverse = np.diag(w_inverse)

        # Matrix A: transpose(vH) diag(w_inverse) transpose(u)
        A = np.dot(vH.T, np.dot(diag_mat_w_inverse, u.T))

        # Solve linear equation to calculate broadening function
        broadening_function = np.dot(A, spectrum_truncated.T)
        self.bf = np.ravel(broadening_function)
        return self.bf

    def smooth(self):
        """
        Smoothes a calculated broadening function by convolving with a gaussian function.
        """
        if self.bf is None:
            raise TypeError('self.bf is None. self.solve() must be run prior to smoothing the broadening function.')
        gaussian = np.exp(-0.5 * (self.velocity/self.smooth_sigma)**2)
        gaussian /= np.sum(gaussian)
        self.bf_smooth = fftconvolve(self.bf, gaussian, mode='same')
        return self.bf_smooth

    def fit_rotational_profile(self, ifitparams:InitialFitParameters,
                               fitting_routine=fiting_routine_rotational_broadening_profile):
        """
        Fits the broadening function with a rotational broadening profile by calling a fitting routine provided.
        The routine must include all essential parts of the fitting procedure.
        :param ifitparams:          an object holding the initial fit parameters:
               vsini:               float, guess for the v sin(i) parameter
               limbd_coef:          float, a calculated limb darkening coefficient for the star.
                                         Default routine will not fit this parameter.
               velocity_fit_width:  float, how far out the fitting routine should include data-points for the fit.
               spectral_resolution: float/int, the resolution of the spectrograph used for the program spectrum.
        :param fitting_routine:     function, the fitting routine used. Default routine fits 1 rotational broadening
                                    profile to the data following the model provided by Kaluzny 2006: "Eclipsing
                                    Binaries in the Open Cluster NGC 2243 II. Absolute Properties of NV CMa".
        :return (fit, model):       fit: lmfit.MinimizerResult of the performed fit.
                                    model: np.ndarray, model values of the broadening function according to the fit.
        """
        if self.bf_smooth is None:
            raise TypeError('self.bf_smooth. self.smooth() must be run prior to fitting')

        self.fit, self.model_values = fitting_routine(self.velocity, self.bf_smooth, ifitparams, self.smooth_sigma,
                                                      self.dv)
        # TODO: figure out why vsini_guess needs to be provided
        return self.fit, self.model_values



