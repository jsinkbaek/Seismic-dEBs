"""
First edition May 02/2021.
@author Jeppe Sinkbæk Thomsen, Master's thesis studen at Aarhus University.
Supervisor: Karsten Frank Brogaard.

This is a collection of functions that form a routine to perform spectral separation of detached eclipsing binaries
with a giant component and a main sequence component. The routine is adapted from multiple scripts, an IDL script by
Karsten Frank Brogaard named "dis_real_merged_8430105_2021.pro", and a similar python script "spec_8430105_bf.py" by the
same author. Both follows the formula layout of the article:
'Separation of composite spectra: the spectroscopic detecton of an eclipsing binary star'
        by J.F. Gonzalez and H. Levato ( A&A 448, 283-292(2006) )

Other module files which this code uses are also adapted from other sources, including the shazam library for the
SONG telescope (written by Emil Knudstrup).
However, multiple people have been over shazam.py, including Karsten Frank Brogaard and Frank Grundahl, and it itself is
based on previous implementations by J. Jessen Hansen and others.
"""

from RV.library.calculate_radial_velocities import *
from RV.library.broadening_function_svd import *
from RV.library.rotational_broadening_function_fitting import get_fit_parameter_values
from copy import copy
from joblib import Parallel, delayed
from RV.library.initial_fit_parameters import InitialFitParameters


def shift_spectrum(flux, radial_velocity_shift, delta_v):
    shift = int(radial_velocity_shift * delta_v)
    return np.roll(flux, shift)


def shifted_mean_spectrum(flux_collection, radial_velocity_shift_collection, delta_v):
    n_spectra = flux_collection[0, :].size
    mean_flux = np.zeros((flux_collection[:, 0].size, ))
    for i in range(0, n_spectra):
        mean_flux += shift_spectrum(flux_collection[:, i], radial_velocity_shift_collection[i], delta_v)
    mean_flux = mean_flux / n_spectra
    return mean_flux


def separate_component_spectra(flux_collection, radial_velocity_collection_A, radial_velocity_collection_B, delta_v,
                               convergence_limit, max_iterations=10):
    """
    Assumes that component A is the dominant component in the spectrum.
    :param flux_collection:               np.ndarray shape (datasize, nspectra) of all the observed spectra
    :param radial_velocity_collection_A:  np.ndarray shape (nspectra, ) of radial velocity values for component A
    :param radial_velocity_collection_B:  np.ndarray shape (nspectra, ) of radial velocity values for component B
    :param delta_v:                       float, the sampling size of the spectra in velocity space
    :param convergence_limit:             float, the precision needed to break while loop
    :param max_iterations:                int, maximum number of allowed iterations before breaking loop

    :return separated_flux_A, separated_flux_B:   the separated and meaned total component spectra of A and B.
    """
    n_spectra = flux_collection[0, :].size
    separated_flux_B = np.zeros((flux_collection[:, 0].size,))  # Set to 0 before iteration
    separated_flux_A = np.zeros((flux_collection[:, 0].size,))

    iteration_counter = 0
    while True:
        RMS_values_A = -separated_flux_A
        RMS_values_B = -separated_flux_B
        iteration_counter += 1
        separated_flux_A = np.zeros((flux_collection[:, 0].size,))
        for i in range(0, n_spectra):
            rvA = radial_velocity_collection_A[i]
            rvB = radial_velocity_collection_B[i]
            shifted_flux_A = shift_spectrum(flux_collection[:, i], -rvA, delta_v)
            separated_flux_A += shifted_flux_A - shift_spectrum(separated_flux_B, rvB - rvA, delta_v)
        separated_flux_A = separated_flux_A / n_spectra

        separated_flux_B = np.zeros((flux_collection[:, 0].size,))
        for i in range(0, n_spectra):
            rvA = radial_velocity_collection_A[i]
            rvB = radial_velocity_collection_B[i]
            shifted_flux_B = shift_spectrum(flux_collection[:, i], -rvB, delta_v)
            separated_flux_B += shifted_flux_B - shift_spectrum(separated_flux_A, rvA - rvB, delta_v)
        separated_flux_B = separated_flux_B / n_spectra

        RMS_values_A += separated_flux_A
        RMS_values_B += separated_flux_B
        RMS_A = np.sum(RMS_values_A**2)/RMS_values_A.size
        RMS_B = np.sum(RMS_values_B**2)/RMS_values_B.size
        if RMS_A < convergence_limit and RMS_B < convergence_limit:
            print(f'Separate Component Spectra: Convergence limit of {convergence_limit} successfully reached in '
                  f'{iteration_counter} iterations. \nReturning last separated spectra.')
        elif iteration_counter >= max_iterations:
            warnings.warn(f'Warning: Iteration limit of {max_iterations} reached without reaching convergence limit'
                          f' of {convergence_limit}. \nCurrent RMS_A: {RMS_A}. RMS_B: {RMS_B} \n'
                          'Returning last separated spectra.')

    return separated_flux_A, separated_flux_B


def recalculate_RVs(flux_collection_inverted, separated_flux_A, separated_flux_B, RV_collection_A,
                                  RV_collection_B, flux_templateA_inverted, flux_templateB_inverted, delta_v,
                                  ifitparamsA:InitialFitParameters, ifitparamsB:InitialFitParameters,
                                  broadening_function_smooth_sigma=4.0, bf_velocity_span=381):
    n_spectra = flux_collection_inverted[0, :].size
    v_span = bf_velocity_span
    BRsvd_template_A = BroadeningFunction(flux_collection_inverted[:, 0], flux_templateA_inverted, v_span, delta_v)
    BRsvd_template_B = BroadeningFunction(flux_collection_inverted[:, 0], flux_templateB_inverted, v_span, delta_v)
    BRsvd_template_A.smooth_sigma = broadening_function_smooth_sigma
    BRsvd_template_B.smooth_sigma = broadening_function_smooth_sigma

    for i in range(0, n_spectra):
        corrected_flux_A = flux_collection_inverted[:, i] -shift_spectrum(separated_flux_B, RV_collection_B[i], delta_v)
        corrected_flux_b = flux_collection_inverted[:, i] -shift_spectrum(separated_flux_A, RV_collection_A[i], delta_v)

        RV_collection_A[i], _ = radial_velocity_single_component(corrected_flux_A, BRsvd_template_A, ifitparamsA)
        RV_collection_B[i], _ = radial_velocity_single_component(corrected_flux_b, BRsvd_template_B, ifitparamsB)
    return RV_collection_A, RV_collection_B


def spectral_separation_routine(flux_collection_inverted, flux_templateA_inverted, flux_templateB_inverted, delta_v,
                                ifitparamsA:InitialFitParameters, ifitparamsB:InitialFitParameters,
                                broadening_function_smooth_sigma=4.0, number_of_parallel_jobs=4,
                                bf_velocity_span=381, convergence_limit=1E-5, iteration_limit=10):
    # Calculate initial guesses for radial velocity values
    RVs = radial_velocities_of_multiple_spectra(flux_collection_inverted, flux_templateA_inverted, delta_v,
                                                ifitparamsA, ifitparamsB, broadening_function_smooth_sigma,
                                                number_of_parallel_jobs, bf_velocity_span)
    RV_collection_A, RV_collection_B = RVs[0], RVs[1]

    # Iterative loop that repeatedly separates the spectra from each other in order to calculate new RVs (Gonzales 2005)
    iterations = 0
    while True:
        separated_flux_A, separated_flux_B = separate_component_spectra(flux_collection_inverted, RV_collection_A,
                                                                        RV_collection_B, delta_v, convergence_limit)

        RV_collection_A, RV_collection_B \
            = recalculate_RVs(flux_collection_inverted, separated_flux_A, separated_flux_B, RV_collection_A,
                              RV_collection_B, flux_templateA_inverted, flux_templateB_inverted, delta_v, ifitparamsA,
                              ifitparamsB, broadening_function_smooth_sigma, bf_velocity_span)
        iterations += 1
        if iterations >= iteration_limit:
            break
    return RV_collection_A, RV_collection_B, separated_flux_A, separated_flux_B

