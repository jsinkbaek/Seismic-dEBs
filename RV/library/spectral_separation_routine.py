"""
First edition May 02/2021.
@author Jeppe Sinkbæk Thomsen, Master's thesis studen at Aarhus University.
Supervisor: Karsten Frank Brogaard.

This is a collection of functions that form a routine to perform spectral separation of detached eclipsing binaries
with a giant component and a main sequence component. The routine is adapted from multiple scripts, an IDL script by
Karsten Frank Brogaard named "dis_real_merged_8430105_2021.pro", and a similar python script "spec_8430105_bf.py" by the
same author. Both follows the formula layout from the article:
'Separation of composite spectra: the spectroscopic detecton of an eclipsing binary star'
        by J.F. Gonzalez and H. Levato ( A&A 448, 283-292(2006) )

Other module files which are part of this code are also adapted from other sources, including the shazam library for the
SONG telescope (written by Emil who has not stated their full name in their library).
"""

from RV.library.broadening_function_svd import *
from RV.library.rotational_broadening_function_fitting import get_fit_parameter_values
from copy import copy
from joblib import Parallel, delayed


def spectral_separation_stepper(number_of_parallel_jobs, broadening_function_svd_giant:BroadeningFunction,
                                broadening_function_svd_ms:BroadeningFunction, flux_collection_inverted, bf_models_ms,
                                vsini_guess_giant, vsini_guess_ms, spectral_resolution_giant, spectral_resolution_ms,
                                velocity_fit_width_giant, velocity_fit_width_ms, limbd_coef_giant,
                                limbd_coef_ms=None):
    """
    Performs one full iteration of the iterative spectral separation process that fits a single broadening function to
    attempt to capture the giant stellar spectrum (the most significant signal), subtract it and then attempt to capture
    the main-sequence stellar spectrum. It does this for all the inputted spectra in parallel, and then supplies the
    resulting broadening function fit and model values back, so that a new iteration can be performed.
    :param number_of_parallel_jobs:         int, how many jobs (cpu threads (or cores maybe) and memory copies) should
                                            be utilized?
    :param broadening_function_svd_giant:   the BroadeningFunction object to use as a template for the giant star.
                                            It is most important that the template_spectrum and svd, smooth_sigma and
                                            velocities attributes are all correct.
    :param broadening_function_svd_ms:      Same as before, but for the main sequence star.
    :param flux_collection_inverted:        Flux values for all the spectra. Should be a np.ndarray of shape (x, n),
                                            where x is the dataset size, and n is the number of spectra.
    :param bf_models_ms:                    np.ndarray. Model values of the main sequence broadening function from
                                            previous iterations. Should be shape (y, n), where y is the length of each
                                            model, and n is the number of models (and therefore the number of spectra).
    :param vsini_guess_giant:               float, guess or fit value from previous iteration for the v sin(i) parameter
                                            for giant.
    :param vsini_guess_ms:                  same as before, but for main sequence star
    :param spectral_resolution_giant:       float or int, the resolution of the spectrograph to use in subprocesses.
    :param spectral_resolution_ms:          same as before, but for the main sequence star (set separately in case one
                                            wants to assume a lower resolution for the dataset when fitting for the
                                            smaller peak).
    :param velocity_fit_width_giant:        float, half width to include around peak when fitting the giant star.
    :param velocity_fit_width_ms:           same but for main sequence fit
    :param limbd_coef_giant:                float, the linear limb darkening coefficient of the giant star.
    :param limbd_coef_ms:                   float, the linear limb darkening coefficient of the main sequence star.

    :return iteration_result:     np.ndarray of shape (n, ), with tuples
                                  (fit_broadening_function_giant, model_values_bf_giant,
                                   fit_broadening_function_ms, model_values_bf_ms)

                                  fit is an lmfit.minimize.MinimizerResult which has the fit parameters and details.
                                  model_values are model values of the broadening function fit sampled at the same
                                  velocities as the template BroadeningFunction objects.
    """
    parallel_process_arguments = (
        broadening_function_svd_giant, broadening_function_svd_ms, flux_collection_inverted, bf_models_ms,
        vsini_guess_giant, vsini_guess_ms, spectral_resolution_giant, spectral_resolution_ms,
        velocity_fit_width_giant, velocity_fit_width_ms, limbd_coef_giant, limbd_coef_ms
    )
    result_parallel_process = Parallel(n_jobs=number_of_parallel_jobs)\
        (delayed(__process_input)(i, parallel_process_arguments) for i in range(0, flux_collection_inverted[0, :].size))

    iteration_result = np.empty((flux_collection_inverted[0, :].size, ), dtype=tuple)
    for i in range(0, len(result_parallel_process)):
        iteration_result[i] = tuple(result_parallel_process[i])

    return iteration_result


def __process_input(i, broadening_function_svd_giant:BroadeningFunction, broadening_function_svd_ms:BroadeningFunction,
                    flux_collection_inverted, bf_models_ms, vsini_guess_giant, vsini_guess_ms,
                    spectral_resolution_giant, spectral_resolution_ms, velocity_fit_width_giant, velocity_fit_width_ms,
                    limbd_coef_giant, limbd_coef_ms=None):
    """
    The process to be run in parallel in spectral_separation_stepper(). See the docstring in that function for details
    about the intended purpose and parameter explanations.
    """
    # Make copy of objects for this loop iteration
    BFsvd_G, BFsvd_MS = copy(broadening_function_svd_giant), copy(broadening_function_svd_ms)

    # Select vsini
    if isinstance(vsini_guess_ms, np.ndarray):
        vsini_guess_ms = vsini_guess_ms[i]
        vsini_guess_giant = vsini_guess_giant[i]

    # Set program spectrum to current one
    BFsvd_G.spectrum = flux_collection_inverted[:, i]
    BFsvd_MS.spectrum = BFsvd_G.spectrum

    # Create broadening function for Giant star
    BFsvd_G.solve()
    BFsvd_G.bf = BFsvd_G.bf - bf_models_ms[:, i]
    BFsvd_G.smooth()

    # Fit broadening function peak for giant
    fit_bf_G, model_bf_G = BFsvd_G.fit_rotational_profile(vsini_guess_giant, limbd_coef_giant, velocity_fit_width_giant,
                                                          spectral_resolution_giant)
    # Create broadening function for MS star
    BFsvd_MS.bf = BFsvd_G.bf - model_bf_G
    BFsvd_MS.smooth()

    # Fit broadening function peak for MS
    if limbd_coef_ms is None:
        limbd_coef_ms = limbd_coef_giant
    fit_bf_MS, model_bf_MS = BFsvd_MS.fit_rotational_profile(vsini_guess_ms, limbd_coef_ms, velocity_fit_width_ms,
                                                             spectral_resolution_ms)

    return [fit_bf_G, model_bf_G, fit_bf_MS, model_bf_MS]


def spectral_separation_stepper_2(BFsvd_giant:BroadeningFunction, BFsvd_ms:BroadeningFunction, model_values_ms,
                                  vsini_guess_giant, vsini_guess_ms, spectral_resolution_giant,
                                  spectral_resolution_ms, velocity_fit_width_giant, velocity_fit_width_ms,
                                  limbd_coef_giant, limbd_coef_ms=None):
    """

    :param BFsvd_giant:
    :param BFsvd_ms:
    :param model_values_ms:
    :param vsini_guess_giant:
    :param vsini_guess_ms:
    :param spectral_resolution_giant:
    :param spectral_resolution_ms:
    :param velocity_fit_width_giant:
    :param velocity_fit_width_ms:
    :param limbd_coef_giant:
    :param limbd_coef_ms:
    :return:
    """
    # TODO: Implement radial velocity guessing in fitting routine so it can be input from here
    # Create Broadening Function for Giant Star
    BFsvd_giant.solve()
    BFsvd_giant.bf = BFsvd_giant.bf - model_values_ms       # remove last iteration MS broadening function
    BFsvd_giant.smooth()

    # Fit rotational broadening function profile to Giant broadening function peak
    fit_bf_giant, model_values_giant = BFsvd_giant.fit_rotational_profile(vsini_guess_giant, limbd_coef_giant,
                                                                          velocity_fit_width_giant,
                                                                          spectral_resolution_giant)

    # Create Broadening Function for Main Sequence Star
    BFsvd_ms.bf = BFsvd_giant.bf - model_values_giant
    BFsvd_ms.smooth()

    # Fit rotational broadening function profile to MS broadening function peak
    if limbd_coef_ms is None:
        limbd_coef_ms = limbd_coef_giant
    fit_bf_ms, model_values_ms = BFsvd_ms.fit_rotational_profile(vsini_guess_ms, limbd_coef_ms, velocity_fit_width_ms,
                                                                 spectral_resolution_ms)
    return fit_bf_giant, model_values_giant, fit_bf_ms, model_values_ms


def spectral_separation_driver(flux_collection_inverted, flux_template_inverted, delta_v,
                               vsini_guess_giant, vsini_guess_ms, spectral_resolution_giant, spectral_resolution_ms,
                               velocity_fit_width_giant, velocity_fit_width_ms, limbd_coef_giant, limbd_coef_ms=None,
                               broadening_function_smooth_sigma=4.0, number_of_parallel_jobs=4,
                               broadening_function_span=381, convergence_limit=1E-5, iteration_limit=10):
    """
    This function prepares the initial steps of the broadening function singular value decomposition, drives
    spectral_separation_stepper() in a while loop (which fits the broadening function of the two components
    iteratively), evaluates its results, and returns RVs of the components if an RMS value passes below a convergence
    limit or an iteration limit is reached. Only parameters not described in spectral_separation_iteration() will be
    explained below.
    :param flux_collection_inverted:  Inverted flux values for all the spectra. Should be a np.ndarray of shape (x, n),
                                            where x is the dataset size, and n is the number of spectra. np.ndarray
    :param flux_template_inverted:    Inverted flux values (1-normalized flux) for the template spectrum which the SVD
                                            is created from. np.ndarray
    :param delta_v:                   float, resolution of the flux values in velocity space (they must be resampled to
                                            equi-spaced values in velocity space prior to the function call)
    :param vsini_guess_giant:
    :param vsini_guess_ms:
    :param spectral_resolution_giant:
    :param spectral_resolution_ms:
    :param velocity_fit_width_giant:
    :param velocity_fit_width_ms:
    :param limbd_coef_giant:
    :param limbd_coef_ms:
    :param broadening_function_smooth_sigma: float, sigma value for the gaussian smoothing of the broadening function
    :param number_of_parallel_jobs:
    :param broadening_function_span:   int, span of the broadening function that is to be created (size of the design
                                            matrix)
    :param convergence_limit:          float, the limit in change in RMS value before results are returned.
    :param iteration_limit:            int, the maximum allowed number of iterations before results are returned,
                                            regardless of RMS value
    :return (RV_giant, RV_ms):         a tuple of 2 np.ndarray objects, holding results from the spectral separation.
                                            RV_giant: the found radial velocities for the giant component,
                                            for each spectrum.
                                            RV_ms: same, but for the main sequence component.
    """
    span = broadening_function_span

    BFsvd_template_giant = BroadeningFunction(flux_collection_inverted[:, 0], flux_template_inverted, span, delta_v)
    BFsvd_template_giant.smooth_sigma = broadening_function_smooth_sigma
    BFsvd_template_ms = copy(BFsvd_template_giant)

    # Set initial main sequence broadening function model to 0
    model_values_ms = np.zeros((BFsvd_template_ms.velocity.size, flux_collection_inverted[0, :].size))
    RV_giant = np.zeros((flux_collection_inverted[0, :].size,))
    RV_ms = np.zeros((flux_collection_inverted[0, :].size,))
    vsini_giant = np.empty((flux_collection_inverted[0, :].size,))
    vsini_ms = np.empty((flux_collection_inverted[0, :].size,))

    iteration_counter = 0

    while True:
        RMS_vals_giant = -RV_giant
        RMS_vals_ms    = -RV_ms

        iteration_results = spectral_separation_stepper(number_of_parallel_jobs, BFsvd_template_giant,
                                                        BFsvd_template_ms, flux_collection_inverted, model_values_ms,
                                                        vsini_guess_giant, vsini_guess_ms, spectral_resolution_giant,
                                                        spectral_resolution_ms, velocity_fit_width_giant,
                                                        velocity_fit_width_ms, limbd_coef_giant, limbd_coef_ms)
        # TODO: Ask if these vsini are supposed to be independent for both the two components and different observations

        for i in range(0, len(iteration_results)):
            result = iteration_results[i]
            (fit_giant, model_values_giant, fit_ms, model_values_ms) = result
            _, RV_giant[i], vsini_giant[i], _, _, _ = get_fit_parameter_values(fit_giant.params)
            _, RV_ms[i], vsini_ms[i], _, _, _ = get_fit_parameter_values(fit_ms.params)

        vsini_guess_giant = vsini_giant
        vsini_guess_ms = vsini_ms

        RMS_vals_giant += RV_giant
        RMS_vals_ms += RV_ms

        iteration_counter += 1
        if np.sum(RMS_vals_giant**2)/RMS_vals_giant.size < convergence_limit:
            print(f'Separation Driver: Convergence limit of {convergence_limit} successfully reached in '
                  f'{iteration_counter} iterations. \nReturning last RV results.')
            break
        elif iteration_counter >= iteration_limit:
            warnings.warn(f'Warning: Iteration limit of {iteration_limit} reached without reaching convergence limit'
                          f' of {convergence_limit}. \nCurrent RMS: {np.sum(RMS_vals_giant**2)/RMS_vals_giant.size}. \n'
                          'Returning last RV results.')
            break
    return RV_giant, RV_ms


def spectral_separation_driver_2(flux_inverted, flux_template_inverted, delta_v, vsini_guess_giant, vsini_guess_ms,
                                 spectral_resolution_giant, spectral_resolution_ms, velocity_fit_width_giant,
                                 velocity_fit_width_ms, limbd_coef_giant, limbd_coef_ms=None,
                                 broadening_function_smooth_sigma=4.0, broadening_function_span=381,
                                 convergence_limit=1E-5, iteration_limit=10, spectrum_name=None):
    span = broadening_function_span
    BFsvd_template_giant = BroadeningFunction(flux_inverted, flux_template_inverted, span, delta_v)
    BFsvd_template_giant.smooth_sigma = broadening_function_smooth_sigma
    BFsvd_template_ms = copy(BFsvd_template_giant)

    # Set initial main sequence broadening function model to 0
    model_values_ms = np.zeros((BFsvd_template_ms.velocity.size, flux_inverted.size))

    RV_giant = 0
    RV_ms = 0
    vsini_giant = vsini_guess_giant
    vsini_ms = vsini_guess_ms
    iteration_counter = 0

    while True:
        RMS_giant = -RV_giant
        RMS_ms    = -RV_ms
        iteration_counter += 1

        iteration_result = spectral_separation_stepper_2(BFsvd_template_giant, BFsvd_template_ms, model_values_ms,
                                                         vsini_giant, vsini_ms, spectral_resolution_giant,
                                                         spectral_resolution_ms, velocity_fit_width_giant,
                                                         velocity_fit_width_ms, limbd_coef_giant, limbd_coef_ms)

        fit_giant, model_values_giant, fit_ms, model_values_ms = iteration_result
        _, RV_giant, vsini_giant, _, _, _ = get_fit_parameter_values(fit_giant)
        _, RV_ms, vsini_ms, _, _, _ = get_fit_parameter_values(fit_ms)

        RMS_giant = (RMS_giant + RV_giant)**2
        RMS_ms    = (RMS_ms + RV_ms)**2
        if RMS_giant <= convergence_limit:
            if spectrum_name is not None:
                print(f'Spectrum name: {spectrum_name}')
            print(f'Separation Driver: Convergence limit of {convergence_limit} successfully reached in '
                  f'{iteration_counter} iterations. \nReturning last RV result.')
            break
        elif iteration_counter >= iteration_limit:
            if spectrum_name is not None:
                print(f'Spectrum name: {spectrum_name}')
            warnings.warn(f'Warning: Iteration limit of {iteration_limit} reached without reaching convergence limit'
                          f' of {convergence_limit}. \nCurrent RMS: {RMS_giant}. \n'
                          'Returning last RV results.')
            break
    return RV_giant, RV_ms


def multiple_spectra(flux_collection_inverted, flux_template_inverted, delta_v, vsini_guess_giant, vsini_guess_ms,
                     spectral_resolution_giant, spectral_resolution_ms, velocity_fit_width_giant, velocity_fit_width_ms,
                     limbd_coef_giant, limbd_coef_ms=None, broadening_function_smooth_sigma=4.0,
                     number_of_parallel_jobs=4, broadening_function_span=381, convergence_limit=1E-5,
                     iteration_limit=10, spectrum_names=None):

    arguments = (flux_template_inverted, delta_v, vsini_guess_giant, vsini_guess_ms,
                 spectral_resolution_giant, spectral_resolution_ms, velocity_fit_width_giant, velocity_fit_width_ms,
                 limbd_coef_giant, limbd_coef_ms, broadening_function_smooth_sigma, broadening_function_span,
                 convergence_limit, iteration_limit)

    # Create parallel calls to the separation driver
    result_parallel_process = Parallel(n_jobs=number_of_parallel_jobs) \
            (delayed(spectral_separation_driver_2)(i, flux_collection_inverted[:, i], *arguments, spectrum_names[i])
             for i in range(0, flux_collection_inverted[0, :].size))

    RV_collection_giant = np.empty((len(result_parallel_process), ))
    RV_collection_ms    = np.empty((len(result_parallel_process), ))

    # Collect results
    for i in range(0, len(result_parallel_process)):
        RV_collection_giant[i] = result_parallel_process[i][0]
        RV_collection_ms[i]    = result_parallel_process[i][1]

    return RV_collection_giant, RV_collection_ms
