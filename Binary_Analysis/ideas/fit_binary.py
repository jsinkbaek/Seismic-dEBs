import ellc
import numpy as np
import matplotlib.pyplot as plt
from storage_classes import LightCurve, RadialVelocities, LimbDarkeningCoeffs, ParameterValues
from copy import deepcopy
import lmfit
from scipy.optimize import minimize


def fit_binary(
        light_curves: list or LightCurve,
        radial_velocities_A: RadialVelocities, radial_velocities_B: RadialVelocities,
        fit_parameter_names: np.ndarray, initial_values: ParameterValues,
        limbd_A: list or LimbDarkeningCoeffs, limbd_B: list or LimbDarkeningCoeffs,
        limbd_A_rv: LimbDarkeningCoeffs = None, limbd_B_rv: LimbDarkeningCoeffs = None,
        grid_A_rv=None, grid_B_rv=None,
        time_stamps_rv_model: np.ndarray = None,
        rvA_timestamp_mask: np.ndarray = None, rvB_timestamp_mask: np.ndarray = None,
        verbose=1, fit_bounds: tuple = None, tol: float = None
):
    """
    :param light_curves:
    :param radial_velocities_A:
    :param radial_velocities_B:
    :param fit_parameter_names:      array of strings. Indicates the names of the parameters to fit.
    :param initial_values:
    :return:
    """
    if time_stamps_rv_model is None:
        time_stamps_rv_model = radial_velocities_A.times

    x0 = np.empty((fit_parameter_names.size, ))
    for i in range(0, x0.size):
        x0[i] = getattr(initial_values, fit_parameter_names[i])

    all_parameters = deepcopy(initial_values)

    args = (
        fit_parameter_names, all_parameters, light_curves, radial_velocities_A, radial_velocities_B,
        time_stamps_rv_model, limbd_A, limbd_B, limbd_A_rv, limbd_B_rv, grid_A_rv, grid_B_rv, rvA_timestamp_mask,
        rvB_timestamp_mask, verbose
    )
    fit_res = minimize(fit_stepper, x0, args, method='Nelder-Mead', tol=tol, bounds=fit_bounds)
    return fit_res


def fit_stepper(
        fit_parameter_values: np.ndarray,
        fit_parameter_names: np.ndarray,
        current_parameters: ParameterValues,
        light_curves: list or LightCurve,
        radial_velocities_A: RadialVelocities,
        radial_velocities_B: RadialVelocities,
        time_stamps_rv_model: np.ndarray,
        limbd_A: list or LimbDarkeningCoeffs,
        limbd_B: list or LimbDarkeningCoeffs,
        limbd_A_rv: LimbDarkeningCoeffs = None,
        limbd_B_rv: LimbDarkeningCoeffs = None,
        grid_A_rv=None,
        grid_B_rv=None,
        rvA_timestamp_mask: np.ndarray = None,
        rvB_timestamp_mask: np.ndarray = None,
        verbose=1
):
    for i in range(0, len(fit_parameter_values)):
        setattr(current_parameters, fit_parameter_names[i], fit_parameter_values[i])

    return evaluate_rv_and_lc(
        light_curves, radial_velocities_A, radial_velocities_B, time_stamps_rv_model, current_parameters, limbd_A,
        limbd_B, limbd_A_rv, limbd_B_rv, grid_A_rv, grid_B_rv, rvA_timestamp_mask, rvB_timestamp_mask, verbose
    )


def evaluate_rv_and_lc(
        light_curves: list or LightCurve,
        radial_velocities_A: RadialVelocities,
        radial_velocities_B: RadialVelocities,
        time_stamps_rv_model: np.ndarray,
        params: ParameterValues,
        limbd_A: list or LimbDarkeningCoeffs,
        limbd_B: list or LimbDarkeningCoeffs,
        limbd_A_rv: LimbDarkeningCoeffs = None,
        limbd_B_rv: LimbDarkeningCoeffs = None,
        grid_A_rv=None,
        grid_B_rv=None,
        rvA_timestamp_mask: np.ndarray = None,
        rvB_timestamp_mask: np.ndarray = None,
        verbose=1
):
    chisqr_lc = 0
    if isinstance(light_curves, list):
        for i in range(0, len(light_curves)):
            if isinstance(limbd_A, list):
                current_limbd_A = limbd_A[i]
            else:
                current_limbd_A = limbd_A
            if isinstance(limbd_B, list):
                current_limbd_B = limbd_B[i]
            else:
                current_limbd_B = limbd_B

            chisqr_lc += evaluate_light_curve_model(light_curves[i], params, current_limbd_A, current_limbd_B, verbose)
    else:
        chisqr_lc += evaluate_light_curve_model(light_curves, params, limbd_A, limbd_B, verbose)

    chisqr_rv = evaluate_rv_model(
        radial_velocities_A, radial_velocities_B, time_stamps_rv_model, params, grid_A_rv,
        grid_B_rv, limbd_A_rv, limbd_B_rv, rvA_timestamp_mask, rvB_timestamp_mask, verbose
    )

    chisqr = chisqr_lc + chisqr_rv
    return chisqr


def calculate_light_curve(time_stamps: np.ndarray, params: ParameterValues, limbd_A: LimbDarkeningCoeffs,
                          limbd_B: LimbDarkeningCoeffs, verbose=1):
    lc_flux = ellc.lc(
        time_stamps, params.radius_A, params.radius_B, params.sb_ratio, params.inclination, params.third_light,
        params.t_0, params.period, params.semi_major_axis, params.mass_fraction, params.ecosw, params.esinw,
        limbd_A.coeffs, limbd_B.coeffs, params.grav_dark_exponent_A, params.grav_dark_exponent_B,
        params.incl_change_rate, params.apsidal_motion_rate, params.async_rot_factor_A, params.async_rot_factor_B,
        params.hf_A, params.hf_B, params.boosting_factor_A, params.boosting_factor_B, params.heat_reflection_A,
        params.heat_reflection_B, params.proj_obliquity_A, params.proj_obliquity_B, params.vsini_A, params.vsini_B,
        params.exp_time, params.finite_exptime_integration_points, params.gridsize_A, params.gridsize_B,
        limbd_A.ld_mode, limbd_B.ld_mode, params.shape_A, params.shape_B, params.spot_params_A, params.spot_params_B,
        params.exact_grav, verbose
    )
    return lc_flux


def evaluate_light_curve_model(
        light_curve: LightCurve, params: ParameterValues,
        limbd_A: LimbDarkeningCoeffs, limbd_B: LimbDarkeningCoeffs, verbose=1
):
    try:
        model_flux = calculate_light_curve(light_curve.times, params, limbd_A, limbd_B, verbose)
        model_mag = -2.5*np.log10(model_flux)
        difference = light_curve.magnitude - model_mag
        weight = 1/light_curve.mag_err**2
        zp = np.sum(difference*weight)/np.sum(weight)
        chisq = np.sum((difference - zp)**2 * weight)
    except ValueError:
        chisq = 1e20
    return chisq


def calculate_rv_model(
        time_stamps: np.ndarray, params: ParameterValues, grid_A_overwrite=None, grid_B_overwrite=None,
        limbd_A: LimbDarkeningCoeffs = None, limbd_B: LimbDarkeningCoeffs = None, verbose=1
):
    if grid_A_overwrite is not None:
        grid_A = grid_A_overwrite
    else:
        grid_A = params.gridsize_A

    if grid_B_overwrite is not None:
        grid_B = grid_B_overwrite
    else:
        grid_B = params.gridsize_B

    if limbd_A is None:
        ldm_A = None
        ldc_A = None
    else:
        ldm_A = limbd_A.ld_mode
        ldc_A = limbd_A.coeffs

    if limbd_B is None:
        ldm_B = None
        ldc_B = None
    else:
        ldm_B = limbd_B.ld_mode
        ldc_B = limbd_B.coeffs

    rv_A_model, rv_B_model = ellc.rv(
        time_stamps, params.radius_A, params.radius_B, params.sb_ratio, params.inclination, params.t_0,
        params.period, params.semi_major_axis, params.mass_fraction, params.ecosw, params.esinw, ldc_A, ldc_B,
        params.grav_dark_exponent_A, params.grav_dark_exponent_B, params.incl_change_rate, params.apsidal_motion_rate,
        params.async_rot_factor_A, params.async_rot_factor_B, params.hf_A, params.hf_B, params.boosting_factor_A,
        params.boosting_factor_B, params.heat_reflection_A, params.heat_reflection_B, params.proj_obliquity_A,
        params.proj_obliquity_B, params.vsini_A, params.vsini_B, grid_1=grid_A, grid_2=grid_B, ld_1=ldm_A, ld_2=ldm_B,
        shape_1=params.shape_A, shape_2=params.shape_B, spots_1=params.spot_params_A, spots_2=params.spot_params_B,
        flux_weighted=params.flux_weighted_rv, verbose=verbose
    )
    return rv_A_model, rv_B_model


def evaluate_rv_model(
        rv_A: RadialVelocities, rv_B: RadialVelocities, time_stamps: np.ndarray, params: ParameterValues,
        grid_A_overwrite=None, grid_B_overwrite=None, limbd_A: LimbDarkeningCoeffs = None,
        limbd_B: LimbDarkeningCoeffs = None, A_timestamp_mask=None, B_timestamp_mask=None,
        verbose=1
):
    try:
        rv_A_model, rv_B_model = calculate_rv_model(
            time_stamps, params, grid_A_overwrite, grid_B_overwrite, limbd_A, limbd_B, verbose
        )
        if A_timestamp_indices is not None:
            rv_A_model = rv_A_model[A_timestamp_mask]
        if B_timestamp_indices is not None:
            rv_B_model = rv_B_model[B_timestamp_mask]
        difference_A = rv_A.values - rv_A_model
        difference_B = rv_B.values - rv_B_model
        weight_A = 1./rv_A.errors**2
        weight_B = 1./rv_B.errors**2

        zp_A = np.sum(difference_A*weight_A)/np.sum(weight_A)
        zp_B = np.sum(difference_B*weight_B)/np.sum(weight_B)

        chisq_rv = np.sum((difference_A-zp_A)**2 * weight_A) + np.sum((difference_B-zp_B)**2 * weight_B)
    except ValueError:
        chisq = 1e20
    return chisq
