import numpy as np
from fit_binary import fit_binary
from storage_classes import RadialVelocities, LightCurve, ParameterValues, LimbDarkeningCoeffs

time_A, rvA, err_rvA = np.loadtxt('rvA.txt', unpack=True)
time_B, rvB, err_rvB = np.loadtxt('rvB.txt', unpack=True)
RV_A = RadialVelocities(time_A, rvA, err_rvA)
RV_B = RadialVelocities(time_B, rvB, err_rvB)

time_kepler, flux_kepler, err_kepler = np.loadtxt('lc.kepler.txt', unpack=True)
time_tess, flux_tess, err_tess = np.loadtxt('lc.tess.txt', unpack=True)
lc_tess = LightCurve(time_tess, flux_tess, err_tess)
lc_kepler = LightCurve(time_kepler, flux_kepler, err_kepler)

initial_params = ParameterValues(
    radius_A=7.6737, radius_B=0.769115, sb_ratio=1/0.612366, inclination=88.9986,
    t_0=54976.6, period=63.3271, semi_major_axis=86.18042, mass_fraction=1.313978/0.82759, secosw=-0.49718,
    sesinw=0.0975, flux_weighted_rv=False
)

fit_vary_names = np.array(['radius_A', 'radius_B', 'sb_ratio', 'inclination', 't_0', 'semi_major_axis', 'mass_fraction',
                           'secosw', 'sesinw'])

limbd_A_kepler = LimbDarkeningCoeffs('quad', 0.476335, 0.211056)
limbd_B_kepler = LimbDarkeningCoeffs('quad', 0.369030, 0.278022)
limbd_A_tess = LimbDarkeningCoeffs('quad', 0.366335, 0.244559)
limbd_B_tess = LimbDarkeningCoeffs('quad', 0.287742, 0.281417)

fit_res = fit_binary(
    [lc_tess, lc_kepler], RV_A, RV_B, fit_vary_names, initial_params, [limbd_A_tess, limbd_A_kepler],
    [limbd_B_tess, limbd_B_kepler]
)
print(fit_res.values())
