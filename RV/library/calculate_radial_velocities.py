from RV.library.broadening_function_svd import *
from RV.library.rotational_broadening_function_fitting import get_fit_parameter_values
from copy import copy
from joblib import Parallel, delayed
from RV.library.initial_fit_parameters import InitialFitParameters


def radial_velocity_2_components(
        inv_flux, broadening_function_template:BroadeningFunction, ifitparamsA:InitialFitParameters,
        ifitparamsB:InitialFitParameters
):
    BFsvd = copy(broadening_function_template)
    BFsvd.spectrum = inv_flux
    BFsvd.smooth_sigma = ifitparamsA.bf_smooth_sigma

    # Create Broadening Function for star A
    BFsvd.solve()
    BFsvd.smooth()

    # Fit rotational broadening function profile to Giant peak
    fit_A, model_values_A = BFsvd.fit_rotational_profile(ifitparamsA)

    # Create Broadening Function for star B
    bf, bf_smooth = BFsvd.bf, BFsvd.bf_smooth
    BFsvd.smooth_sigma = ifitparamsB.bf_smooth_sigma
    BFsvd.bf = BFsvd.bf - model_values_A        # subtract model for giant
    BFsvd.smooth()

    # Fit rotational broadening function profile for MS peak
    if ifitparamsB.limbd_coef is None:
        ifitparamsB.limbd_coef = ifitparamsA.limbd_coef
    fit_B, model_values_B = BFsvd.fit_rotational_profile(ifitparamsB)

    _, RV_A, _, _, _, _ = get_fit_parameter_values(fit_A.params)
    _, RV_B, _, _, _, _ = get_fit_parameter_values(fit_B.params)
    return (RV_A, RV_B), (model_values_A, fit_A, model_values_B, fit_B), (bf, bf_smooth)


def _pull_results_mspectra_2comp(broadening_function_template: BroadeningFunction, res_par, n_spectra, plot,
                                 spectrum_size):
    RVs_A = np.empty((n_spectra,))
    RVs_B = np.empty((n_spectra,))
    broadening_function_vals = np.empty((broadening_function_template.velocity.size, n_spectra))
    broadening_function_vals_smoothed = np.empty((broadening_function_template.velocity.size, n_spectra))
    model_values_A = np.empty((broadening_function_template.velocity.size, n_spectra))
    model_values_B = np.empty((broadening_function_template.velocity.size, n_spectra))
    for i in range(0, n_spectra):
        RV_values = res_par[i][0]
        models, (bf, bf_smooth) = res_par[i][1], res_par[i][2]
        broadening_function_vals[:, i] = bf
        broadening_function_vals_smoothed[:, i] = bf_smooth
        model_values_A[:, i] = models[0]
        model_values_B[:, i] = models[2]
        if plot:
            plt.figure()
            plt.plot(broadening_function_template.velocity, bf_smooth)
            plt.plot(broadening_function_template.velocity, models[0], 'k--')
            plt.plot(broadening_function_template.velocity, models[2], 'k--')
            plt.show(block=False)
        RVs_A[i], RVs_B[i] = RV_values[0], RV_values[1]
    if plot:
        plt.show(block=True)
    extra_results = (broadening_function_template.velocity, broadening_function_vals, broadening_function_vals_smoothed,
                     model_values_A, model_values_B)
    return RVs_A, RVs_B, extra_results


def _pull_results_mspectra_1comp(broadening_function_template: BroadeningFunction, res_par, n_spectra, plot,
                                 spectrum_size):
    RVs = np.empty((n_spectra, ))
    broadening_function_vals = np.empty((broadening_function_template.velocity.size, n_spectra))
    broadening_function_vals_smoothed = np.empty((broadening_function_template.velocity.size, n_spectra))
    model_values = np.empty((broadening_function_template.velocity.size, n_spectra))
    for i in range(0, n_spectra):
        RVs[i] = res_par[i][0]
        fit_res = res_par[i][1]
        model_values[:, i], broadening_function_vals[:, i] = fit_res[1], fit_res[3]
        broadening_function_vals_smoothed[:, i] = fit_res[4]
        if plot:
            plt.figure()
            plt.plot(broadening_function_template.velocity, broadening_function_vals_smoothed[:, i])
            plt.plot(broadening_function_template.velocity, model_values, 'k--')
            plt.show(block=False)
    if plot:
        plt.show(block=True)
    extra_results = (broadening_function_template.velocity, broadening_function_vals, broadening_function_vals_smoothed,
                     model_values)
    return RVs, extra_results


def radial_velocities_of_multiple_spectra(
        inv_flux_collection: np.ndarray, inv_flux_template: np.ndarray, delta_v: float,
        ifitparamsA:InitialFitParameters, ifitparamsB: InitialFitParameters = None, number_of_parallel_jobs=4,
        plot=False
):
    """
    Calculates radial velocities for two components, for multiple spectra, by fitting two rotational broadening function
    profiles successively to calculated broadening functions. Uses joblib to parallelize the calculations. Calls
    radial_velocity_from_broadening_function() for each spectrum.

    Assumes that both components are well-described by the single template spectrum provided for the broadening function
    calculation.

    :param inv_flux_collection:     np.ndarray shape (:, n_spectra). Collection of inverted fluxes for each spectrum.
    :param inv_flux_template:       np.ndarray shape (:, ). Inverted flux of a template spectrum.
    :param delta_v:                 float. The sampling delta_v of the velocity grid (resolution). Example 1.0 km/s.
    :param ifitparamsA:             InitialFitParameters. Fitting and broadening function parameters for component A.
    :param ifitparamsB:             InitialFitParameters. Fitting and broadening function parameters for component B.
    :param number_of_parallel_jobs: int. Indicates the number of separate processes to spawn with joblib.
    :param plot:                    bool. Indicates whether results should be plotted, with separate figures for each
                                        spectrum.
    :return:    RVs_A, RVs_B, extra_results.
                RVs_A:  np.ndarray shape (n_spectra, ). RV values of component A.
                RVs_B:  np.ndarray shape (n_spectra, ). RV values of component B.
                extra_results: (broadening function velocity values, broadening function values,
                                smoothed broadening function values, fit model values for component A,
                                fit model values for component B).
    """
    n_spectra = inv_flux_collection[0, :].size
    broadening_function_template = BroadeningFunction(inv_flux_collection[:, 0], inv_flux_template,
                                                      ifitparamsA.bf_velocity_span, delta_v)
    broadening_function_template.smooth_sigma = ifitparamsA.bf_smooth_sigma

    # Arguments for parallel job
    if ifitparamsB is not None:
        arguments = (broadening_function_template, ifitparamsA, ifitparamsB)
        calc_function = radial_velocity_2_components
    else:
        arguments = (broadening_function_template, ifitparamsA)
        calc_function = radial_velocity_single_component

    # Create parallel call to calculate radial velocities
    res_par = Parallel(n_jobs=number_of_parallel_jobs)(
        delayed(calc_function)(inv_flux_collection[:, i], *arguments) for i in range(0, n_spectra)
    )

    # Pull results from call
    if ifitparamsB is not None:
        RVs_A, RVs_B, extra_results = _pull_results_mspectra_2comp(
            broadening_function_template, res_par, n_spectra, plot, spectrum_size=inv_flux_collection[:, 0].size
        )
        bf_velocity, bf_vals, bf_vals_smooth, model_vals_A, model_vals_B = extra_results
        return RVs_A, RVs_B, (bf_velocity, bf_vals, bf_vals_smooth, model_vals_A, model_vals_B)
    else:
        RVs, (bf_velocity, bf_vals, bf_vals_smooth, model_vals) = _pull_results_mspectra_1comp(
            broadening_function_template, res_par, n_spectra, plot, spectrum_size=inv_flux_collection[:, 0].size
        )
        return RVs, (bf_velocity, bf_vals, bf_vals_smooth, model_vals)


def radial_velocity_single_component(
        inv_flux: np.ndarray, broadening_function_template: BroadeningFunction, ifitparams: InitialFitParameters
):
    """
    Calculates the broadening function of a spectrum and fits a single rotational broadening function profile to it.
    Needs a template object of the BroadeningFunction with the correct parameters and template spectrum already set. 
    Setup to be of convenient use during the spectral separation routine (see spectral_separation_routine.py).
    
    :param inv_flux:                     np.ndarray. Inverted flux of the program spectrum (e.g. 1-normalized_flux)
    :param broadening_function_template: BroadeningFunction. The template used to calculate the broadening function.
    :param ifitparams:                   InitialFitParameters. Object that stores the fitting parameters needed.
    :return:    RV, (fit, model_values, BFsvd.velocity, BFsvd.bf, BFsvd.bf_smooth)
                RV:                 float, the fitted RV value
                fit:                lmfit.MinimizerResult. The object storing the fit parameters.
                model_values:       np.ndarray. Broadening function values according to the fit.
                BFsvd.velocity:     np.ndarray. Velocity values for the broadening function and model values.
                BFsvd.bf:           np.ndarray. Broadening function values calculated.
                BFsvd.bf_smooth:    np.ndarray. Smoothed broadening function values.
    """
    BFsvd = copy(broadening_function_template)
    BFsvd.spectrum = inv_flux

    # Create Broadening Function
    BFsvd.solve()
    BFsvd.smooth()

    # Fit rotational broadening function profile
    fit, model_values = BFsvd.fit_rotational_profile(ifitparams)

    _, RV, _, _, _, _ = get_fit_parameter_values(fit.params)

    return RV, (fit, model_values, BFsvd.velocity, BFsvd.bf, BFsvd.bf_smooth)
