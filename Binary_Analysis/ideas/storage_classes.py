import numpy as np


class LimbDarkeningCoeffs:
    def __init__(self, LD_mode, *args):
        """

        :param LD_mode: str, defines limb-darkening law used. Can be one of the following:
                        None, lin, quad, sing, claret, log, sqrt, exp, power-2, mugrid
                        See ellc lc.py for details.
        :param args:    float or 1 str. Gives the limb darkening coefficients, or a string
                        detailing how ellc should calculate it.
        """
        self.ld_mode = LD_mode
        self.coeffs = np.array(args)


class RadialVelocities:
    def __init__(self, time_values: np.ndarray, rv_values: np.ndarray, error_values: np.ndarray):
        self.times = time_values
        self.values = rv_values
        self.errors = error_values


class LightCurve:
    def __init__(self, time_values: np.ndarray, flux_values: np.ndarray, error_values: np.ndarray):
        self.times = time_values
        self.flux = flux_values
        self.error = error_values
        self.magnitude = - 2.5*np.log10(flux_values)
        self.mag_err = np.abs(-2.5/np.log(10) * (error_values/flux_values))


class ParameterValues:
    def __init__(
            self, radius_A, radius_B, surface_brightness_ratio, inclination,
            third_light=0,
            t_0=0, period=10,
            semi_major_axis=None,
            mass_fraction_q=1.0,
            sqrte_cosw=None,
            sqrte_sinw=None,
            grav_dark_exponent_A=None, grav_dark_exponent_B=None,
            inclination_change_rate=None, apsidal_motion_rate=None,
            async_rot_factor_A=1.0, async_rot_factor_B=1.0,
            hf_2love_number_A=1.5, hf_2love_number_B=1.5,
            doppler_boosting_factor_A=None,
            doppler_boosting_factor_B=None,
            heat_reflection_model_A=None, heat_reflection_model_B=None,
            projected_obliquity_A=None, projected_obliquity_B=None,
            vsini_A=None, vsini_B=None,
            exposure_time=None,
            finite_exptime_integration_points=None,
            gridsize_A="default", gridsize_B="default",
            stellar_shape_A="sphere", stellar_shape_B="sphere",
            spot_parameters_A=None, spot_parameters_B=None,
            exact_grav=False,
            flux_weighted_rv=True
    ):
        """
        See ellc lc.py for descriptions of the individual parameters:
        https://github.com/pmaxted/ellc/blob/master/lc.py
        Limb darkening parameters are handled separately to make way for multiple different light curves.
        """
        self.radius_A = radius_A
        self.radius_B = radius_B
        self.sb_ratio = surface_brightness_ratio
        self.inclination = inclination
        self.third_light = third_light
        self.t_0 = t_0
        self.period = period
        self.semi_major_axis = semi_major_axis
        self.mass_fraction = mass_fraction_q
        self.ecosw = sqrte_cosw
        self.esinw = sqrte_sinw
        self.grav_dark_exponent_A = grav_dark_exponent_A
        self.grav_dark_exponent_B = grav_dark_exponent_B
        self.incl_change_rate = inclination_change_rate
        self.apsidal_motion_rate = apsidal_motion_rate
        self.async_rot_factor_A = async_rot_factor_A
        self.async_rot_factor_B = async_rot_factor_B
        self.hf_A = hf_2love_number_A
        self.hf_B = hf_2love_number_B
        self.boosting_factor_A = doppler_boosting_factor_A
        self.boosting_factor_B = doppler_boosting_factor_B
        self.heat_reflection_A = heat_reflection_model_A
        self.heat_reflection_B = heat_reflection_model_B
        self.proj_obliquity_A = projected_obliquity_A
        self.proj_obliquity_B = projected_obliquity_B
        self.vsini_A = vsini_A
        self.vsini_B = vsini_B
        self.exp_time = exposure_time
        self.finite_exptime_integration_points = finite_exptime_integration_points
        self.gridsize_A = gridsize_A
        self.gridsize_B = gridsize_B
        self.shape_A = stellar_shape_A
        self.shape_B = stellar_shape_B
        self.spot_params_A = spot_parameters_A
        self.spot_params_B = spot_parameters_B
        self.exact_grav = exact_grav
        self.flux_weighted_rv = flux_weighted_rv

