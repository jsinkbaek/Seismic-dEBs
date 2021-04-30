"""
First edition on April 30, 2021.
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
import lmfit
import scipy.constants as scc


def rotational_broadening_function_profile(velocities, amplitude, radial_velocity_cm, vsini, gaussian_width,
                                           continuum_constant, limbd_coef):
    """
    Calculates a theoretical broadening function profile based on the one described in
    Kaluzny 2006: Eclipsing Binaries in the Open Cluster NGC 2243 II. Absolute Properties of NV CMa.
    Convolves it with a gaussian function to create a rotational broadening function profile.
    :param velocities:          np.ndarray, velocities to calculate profile for.
    :param amplitude:           float, normalization constant
    :param radial_velocity_cm:  float, radial velocity of the centre of mass of the star
    :param vsini:               float, linear velocity of the equator of the rotating star times sin(inclination)
    :param gaussian_width:      float, width of the gaussian broadening function that the profile will be folded with
    :param continuum_constant:  float, the continuum level
    :param limbd_coef:          float, linear limb darkening coefficient of the star.
    :return rot_bf_profile:     np.ndarray, the profile at the given velocities
    """
    n = velocities.size
    broadening_function_values = np.ones(n) * continuum_constant

    # The "a" coefficient of the profile
    a = (velocities - radial_velocity_cm) / vsini

    # Create bf function values
    mask = (np.abs(a) < 1.0)        # Is this to sort out bad values or down-weigh them in some way?
    broadening_function_values[mask] += amplitude*((1-limbd_coef)*np.sqrt(1.0-a[mask]**2) + 0.25*np.pi*(1-a[mask]**2))

    # Create gs function values
    scaled_width = np.sqrt(2*np.pi) * gaussian_width
    gaussian_function_values = np.exp(-0.5* (velocities/gaussian_width)**2) / scaled_width

    # Convolve to get rotationally broadened broadening function
    rot_bf_profile = fftconvolve(broadening_function_values, gaussian_function_values, mode='same')
    return rot_bf_profile


def weight_function(velocities, broadening_function_values, velocity_fit_width):
    peak_idx = np.argmax(broadening_function_values)
    mask = (velocities > velocities[peak_idx] - velocity_fit_width) & \
           (velocities < velocities[peak_idx] + velocity_fit_width + 1)

    weight_function_values = np.zeros(broadening_function_values.size)
    weight_function_values[mask] = 1.0
    return weight_function_values


def get_fit_parameter_values(parameters):
    amplitude = parameters['amplitude'].value
    radial_velocity_cm = parameters['radial_velocity_cm'].value
    vsini = parameters['vsini'].value
    gaussian_width = parameters['gaussian_width'].value
    continuum_constant = parameters['continuum_constant'].value
    limbd_coef = parameters['limbd_coef'].value

    return amplitude, radial_velocity_cm, vsini, gaussian_width, continuum_constant, limbd_coef


def compare_broadening_function_with_profile(parameters, velocities, broadening_function_values,
                                             weight_function_values):
    parameter_vals = get_fit_parameter_values(parameters)

    comparison = broadening_function_values - rotational_broadening_function_profile(velocities, *parameter_vals)
    return weight_function_values * np.abs(comparison)  # TODO: Ask if absolute value comparison is needed


def fiting_routine_rotational_broadening_profile(velocities, broadening_function_values, vsini, gaussian_width,
                                                 limbd_coef, velocity_fit_width, print_report=False,
                                                 compare_func=compare_broadening_function_with_profile):
    params = lmfit.Parameters()
    peak_idx = np.argmax(broadening_function_values)
    params.add('amplitude', value=broadening_function_values[peak_idx])
    params.add('radial_velocity_cm', value=velocities[peak_idx])
    params.add('vsini', value=vsini)
    params.add('gaussian_width', value=gaussian_width, vary=False)
    params.add('continuum_constant', value=0.0)
    params.add('limbd_coef', value=limbd_coef, vary=False)

    weight_function_values = weight_function(velocities, broadening_function_values, velocity_fit_width)
    fit = lmfit.minimize(compare_func, params, args=(velocities, broadening_function_values, weight_function_values),
                         xtol=1E-8, ftol=1E-8, max_nfev=500)
    if print_report:
        print(lmfit.fit_report(fit, show_correl=False))

    parameter_vals = get_fit_parameter_values(fit.params)
    model = rotational_broadening_function_profile(velocities, broadening_function_values, *parameter_vals)

    return fit, model


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
        self.svd = SingularValueDecomposition(template_spectrum, span)
        self.bf = None
        self.bf_smooth = None
        self.smooth_sigma = 5.0
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

    def smooth(self):
        if self.bf is None:
            raise TypeError('self.bf is None. self.solve() must be run prior to smoothing the broadening function.')
        gaussian = np.exp(-0.5 * (self.velocity/self.smooth_sigma)**2)
        gaussian /= np.sum(gaussian)
        self.bf_smooth = fftconvolve(self.bf, gaussian, mode='same')
        return self.bf_smooth

    def fit_rotational_profile(self, vsini_guess, limbd_coef, velocity_fit_width, spectral_resolution=60000,
                               profile=rotational_broadening_function_profile,
                               fitting_routine=fiting_routine_rotational_broadening_profile):
        if self.bf_smooth is None:
            raise TypeError('self.bf_smooth. self.smooth() must be run prior to fitting')

        speed_of_light = scc.c / 1000       # in km/s
        gaussian_width = np.sqrt(((speed_of_light/spectral_resolution)/(2.354*1.))**2 + (self.smooth_sigma)**2)
        # TODO: Ask Karsten about this gaussian width
        fit, model = fitting_routine(self.velocity, self.bf_smooth, vsini_guess, gaussian_width, limbd_coef,
                                     velocity_fit_width)
        # TODO: figure out why vsini_guess needs to be provided




