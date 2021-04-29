"""
First edition on April 29, 2021.
@author Jeppe Sinkbæk Thomsen.

Purpose of this file is broadening function calculation using a Singular Value Decomposition of a template spectrum.
Code is primarily adapted from http://www.astro.utoronto.ca/~rucinski/SVDcookbook.html and
functions in the shazam.py library for the SONG telescope (written by Emil) TODO: find last name
The primary class to create objects from for broadening function calculation is BroadeningFunction. The rest are
convenience classes for that one.
"""

import numpy as np
import scipy.linalg as lg
from scipy.signal import fftconvolve
import warnings


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
        self.mat = self.map()
        self.n = self.vals.size
        self.m = self.span

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

        self.n = n
        self.m = m

        mat = np.zeros(shape=(m, n-m))
        for i in range(0, m+1):
            mat[i, :] = self.vals[m-i:n-i+1]
        return mat


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
        self.u, self.w, self.vH = lg.svd(self.design_matrix, compute_uv=True, full_matrices=False)


class BroadeningFunction:
    def __init__(self, program_spectrum, template_spectrum, span, dv):
        """
        Creates a broadening function object storing all the necessary variables for it.
        Using BroadeningFunction.solve(), the broadening function is found by solving the Singular Value
        Decomposition of a template spectrum.

        :param program_spectrum:  np.ndarray, inverted flux of the program spectrum equi-spaced in velocity space
        :param template_spectrum: np.ndarray, inverted flux of the template spectrum equi-spaced in velocity space
        :param span:              int, span or width (number of elements) of the broadening function design matrix.
        :param dv:                float, the velocity spectrum resolution in km/s

        :var self.spectrum:  same as program_spectrum
        :var self.svd:       a created SingularValueDecomposition of the template spectrum.
        :var self.bf:        None (before self.solve() is run). After, np.ndarray, the calculated broadening function.
        :var self.bf_smooth: None (before self.smooth() is run). After, np.ndarray, smoothed broadening function.
        :var self.velocity:  np.ndarray, velocity values of the broadening function in km/s.
        """
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
        self.svd = SingularValueDecomposition(template_spectrum, span)
        self.bf = None
        self.bf_smooth = None
        self.velocity = -np.arange(-span/2, span/2+1)*dv

    @staticmethod
    def truncate(spectrum, design_matrix):
        m = design_matrix.m
        return spectrum[m/2:-m/2]

    def solve(self):
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

    def smooth(self, sigma=5.0):
        if self.bf is None:
            raise TypeError('self.bf is None. self.solve() must be run prior to smoothing the broadening function.')
        gaussian = np.exp(-0.5 * (self.velocity/sigma)**2)
        gaussian /= np.sum(gaussian)
        self.bf_smooth = fftconvolve(self.bf, gaussian, mode='same')
        return self.bf_smooth

