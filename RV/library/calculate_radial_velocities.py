from RV.library.broadening_function_svd import *
from RV.library.rotational_broadening_function_fitting import get_fit_parameter_values
from copy import copy
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
    fit_giant, model_values_giant = BFsvd.fit_rotational_profile(ifitparamsA)

    # Create Broadening Function for Main Sequence star
    BFsvd.bf = BFsvd.bf - model_values_giant        # subtract model for giant
    BFsvd.smooth()

    # Fit rotational broadening function profile for MS peak
    if ifitparamsB.limbd_coef is None:
        ifitparamsB.limbd_coef = ifitparamsA.limbd_coef
    fit_ms, model_values_ms = BFsvd.fit_rotational_profile(ifitparamsB)

    _, RV_giant, _, _, _, _ = get_fit_parameter_values(fit_giant)
    _, RV_ms, _, _, _, _ = get_fit_parameter_values(fit_ms)
    return (RV_giant, RV_ms), (model_values_giant, fit_ms, model_values_ms)


def radial_velocities_of_multiple_spectra(flux_collection_inverted, flux_template_inverted, delta_v,
                                          ifitparamsA:InitialFitParameters, ifitparamsB:InitialFitParameters,
                                          broadening_function_smooth_sigma=4.0, number_of_parallel_jobs=4,
                                          broadening_function_span=381):

    broadening_function_template = BroadeningFunction(flux_collection_inverted[:, 0], flux_template_inverted,
                                                      broadening_function_span, delta_v)
    broadening_function_template.smooth_sigma = broadening_function_smooth_sigma

    # Arguments for parallel job
    arguments = (flux_template_inverted, broadening_function_template, ifitparamsA, ifitparamsB)

    # Create parallel call to calculate radial velocities
    res_par = Parallel(n_jobs=number_of_parallel_jobs)\
        (delayed(radial_velocity_from_broadening_function)(flux_collection_inverted[:, i], *arguments)
         for i in range(0, flux_collection_inverted[0, :].size))

    # Pull results from call
    RVs_giant = np.empty((flux_collection_inverted[0, :].size, ))
    RVs_ms = np.empty((flux_collection_inverted[0, :].size, ))
    for i in range(0, flux_collection_inverted[0, :].size):
        RV_values = res_par[i][0]
        RVs_giant[i], RVs_ms[i] = RV_values[0], RV_values[1]
    return RVs_giant, RVs_ms


def radial_velocity_single_component(flux_inverted, broadening_function_template:BroadeningFunction,
                                     ifitparams:InitialFitParameters):
    BFsvd = copy(broadening_function_template)
    BFsvd.spectrum = flux_inverted

    # Create Broadening Function
    BFsvd.solve()
    BFsvd.smooth()

    # Fit rotational broadening function profile
    fit, model_values = BFsvd.fit_rotational_profile(ifitparams)

    _, RV, _, _, _, _ = get_fit_parameter_values(fit)

    return RV, (fit, model_values)
