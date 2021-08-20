import numpy as np
from fit_binary import fit_binary
from storage_classes import RadialVelocities, LightCurve, ParameterValues, LimbDarkeningCoeffs

time_A, rvA, err_rvA = np.loadtxt('rvA.txt', unpack=True)
index_B, time_B, rvB, err_rvB = np.loadtxt('rvB.txt', unpack=True)
RV_A = RadialVelocities(time_A, rvA, err_rvA)
RV_B = RadialVelocities(time_B, rvB, err_rvB)

time_kepler, flux_kepler, err_kepler = np.loadtxt('lc.kepler.txt', unpack=True)
time_tess, flux_tess, err_tess = np.loadtxt('lc.tess.txt', unpack=True)
lc_tess = LightCurve(time_tess, flux_tess, err_tess)
lc_kepler = LightCurve(time_kepler, flux_kepler, err_kepler)

sma = 86.18042
radius_A = 7.6737/sma
radius_B = 0.769115/sma
eccentricity = 0.25666
periastron_longitude = 349.119
perilong_rad = 2*np.pi*periastron_longitude/360
se_cosw = np.sqrt(eccentricity) * np.cos(perilong_rad)
se_sinw = np.sqrt(eccentricity) * np.sin(perilong_rad)


jktebop_ephemerides = 54998.23
phase_eclipse_B_infrontof_A = 0.34102
period = 63.3271
t_0 = jktebop_ephemerides + phase_eclipse_B_infrontof_A * period

initial_params = ParameterValues(
    radius_A=radius_A, radius_B=radius_B, sb_ratio=1/0.612366, inclination=88.9986,
    t_0=t_0, period=period, semi_major_axis=sma, mass_fraction=0.82759/1.313978, secosw=se_cosw,
    sesinw=se_sinw, flux_weighted_rv=False, system_rv=11.7
)
print()

fit_vary_names = np.array(['radius_A', 'radius_B', 'sb_ratio', 'inclination', 't_0', 'semi_major_axis', 'mass_fraction',
                           'secosw', 'sesinw', 'system_rv'])
fit_bounds = np.array([(0, 0.15), (0, 0.05), (1.25, 2.0), (80, 89.9), (jktebop_ephemerides, jktebop_ephemerides+period),
                       (80, 100), (0.5, 1.0), (-1.0, 1.0),
                       (-1.0, 1.0), (0.0, 25.0)])

limbd_A_kepler = LimbDarkeningCoeffs('quad', 0.476335, 0.211056)
limbd_B_kepler = LimbDarkeningCoeffs('quad', 0.369030, 0.278022)
limbd_A_tess = LimbDarkeningCoeffs('quad', 0.366335, 0.244559)
limbd_B_tess = LimbDarkeningCoeffs('quad', 0.287742, 0.281417)

index_B = index_B.astype(int)
mask_B = np.zeros_like(time_A, dtype=bool)
mask_B[index_B] = True


fit_res = fit_binary(
    [lc_tess, lc_kepler], RV_A, RV_B, fit_vary_names, initial_params, [limbd_A_tess, limbd_A_kepler],
    [limbd_B_tess, limbd_B_kepler], rvB_timestamp_mask=mask_B, fit_lc_scales=True
)
print(fit_res.x)
