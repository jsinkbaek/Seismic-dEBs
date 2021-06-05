from RV.library.broadening_function_svd import *
from RV.library.rotational_broadening_function_fitting import get_fit_parameter_values
from copy import copy, deepcopy
from joblib import Parallel, delayed
from RV.library.initial_fit_parameters import InitialFitParameters


def radial_velocity_from_broadening_function(flux_inverted, broadening_function_template:BroadeningFunction,
                                             ifitparamsA:InitialFitParameters, ifitparamsB:InitialFitParameters):
    BFsvd = copy(broadening_function_template)
    BFsvd.spectrum = flux_inverted

    # Create Broadening Function for Giant star
    BFsvd.solve()
    BFsvd.smooth()

    # Fit rotational broadening function profile to Giant peak
    fit_A, model_values_A = BFsvd.fit_rotational_profile(ifitparamsA)

    # Create Broadening Function for Main Sequence star
    bf, bf_smooth = BFsvd.bf, BFsvd.bf_smooth
    BFsvd.bf = BFsvd.bf - model_values_A        # subtract model for giant
    BFsvd.smooth()

    # Fit rotational broadening function profile for MS peak
    if ifitparamsB.limbd_coef is None:
        ifitparamsB.limbd_coef = ifitparamsA.limbd_coef
    fit_B, model_values_B = BFsvd.fit_rotational_profile(ifitparamsB)

    _, RV_A, _, _, _, _ = get_fit_parameter_values(fit_A.params)
    _, RV_B, _, _, _, _ = get_fit_parameter_values(fit_B.params)
    return (RV_A, RV_B), (model_values_A, fit_A, model_values_B, fit_B), (bf, bf_smooth)


def radial_velocities_of_multiple_spectra(flux_collection_inverted, flux_template_inverted, delta_v,
                                          ifitparamsA:InitialFitParameters, ifitparamsB:InitialFitParameters,
                                          broadening_function_smooth_sigma=4.0, number_of_parallel_jobs=4,
                                          bf_velocity_span=381, plot=False):
    n_spectra = flux_collection_inverted[0, :].size
    broadening_function_template = BroadeningFunction(flux_collection_inverted[:, 0], flux_template_inverted,
                                                      bf_velocity_span, delta_v)
    broadening_function_template.smooth_sigma = broadening_function_smooth_sigma

    # Arguments for parallel job
    arguments = (broadening_function_template, ifitparamsA, ifitparamsB)

    # Create parallel call to calculate radial velocities
    res_par = Parallel(n_jobs=number_of_parallel_jobs)\
        (delayed(radial_velocity_from_broadening_function)(flux_collection_inverted[:, i], *arguments)
         for i in range(0, n_spectra))

    # Pull results from call
    RVs_A = np.empty((n_spectra, ))
    RVs_B = np.empty((n_spectra, ))
    broadening_function_vals = np.empty((broadening_function_template.velocity.size, n_spectra))
    broadening_function_vals_smoothed = np.empty((broadening_function_template.velocity.size, n_spectra))
    model_values_A = np.empty((broadening_function_template.velocity.size, n_spectra))
    model_values_B = np.empty((broadening_function_template.velocity.size, n_spectra))
    for i in range(0, flux_collection_inverted[0, :].size):
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
        plt.show()
    extra_results = (broadening_function_template.velocity, broadening_function_vals, broadening_function_vals_smoothed,
                     model_values_A, model_values_B)
    return RVs_A, RVs_B, extra_results


def radial_velocity_single_component(flux_inverted, broadening_function_template:BroadeningFunction,
                                     ifitparams:InitialFitParameters):
    BFsvd = copy(broadening_function_template)
    BFsvd.spectrum = flux_inverted

    # Create Broadening Function
    BFsvd.solve()
    BFsvd.smooth()

    # Fit rotational broadening function profile
    fit, model_values = BFsvd.fit_rotational_profile(ifitparams)

    _, RV, _, _, _, _ = get_fit_parameter_values(fit.params)

    return RV, (fit, model_values, BFsvd.velocity, BFsvd.bf, BFsvd.bf_smooth)
