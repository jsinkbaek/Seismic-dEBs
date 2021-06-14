"""
Script based on RV_from_spectra_kic8430105.py, but for V56 (15162, test spectra).
Stellar values are taken from https://www.aanda.org/articles/aa/pdf/2021/05/aa40911-21.pdf
Brogaard et. al. 2021: Age and helium content of the open cluster NGC 6791 from
multiple eclipsing binary members
"""

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import os
from barycorrpy import get_BC_vel, utc_tdb
import RV.library.spectrum_processing_functions as spf
import warnings
import scipy.constants as scc
import RV.library.AFS_algorithm as afs
import RV.library.calculate_radial_velocities as cRV
from RV.library.initial_fit_parameters import InitialFitParameters
import RV.library.spectral_separation_routine as ssr
from RV.library.linear_limbd_coeff_estimate import estimate_linear_limbd
import matplotlib.pyplot as plt

data_path = 'test_data/'
delta_v = 3.0
estimate_RVb_from_RVa = True
mass_A_estimate = 1.103
mass_B_estimate = 0.974
system_RV_estimate = -46.70
orbital_period_estimate = 10.8219
Teff_A, Teff_B = 5447, 5552
logg_A = (2.78*7.68**2 / 1.31) * (mass_A_estimate / 1.764**2)
logg_B = (4.58*0.7699**2 / 0.8276) * (mass_B_estimate / 1.045**2)
MH_A  , MH_B   = 0.29, 0.29
mTur_A, mTur_B = 2.0, 2.0
wavelength_RV_limit = (5334, 5612)
# limbd_A = estimate_linear_limbd(wavelength_RV_limit, logg_A, Teff_A, MH_A, mTur_A, loc='../../Data/tables/atlasco.dat')
# limbd_B = estimate_linear_limbd(wavelength_RV_limit, logg_B, Teff_B, MH_B, mTur_B, loc='../../Data/tables/atlasco.dat')
ifitpar_A = InitialFitParameters(3.0, 19800, 100, 0.68, smooth_sigma=5.0)
ifitpar_B = InitialFitParameters(3.0, 19800, 40, 0.68, smooth_sigma=5.0)

# # Broadening function and radial velocity parameters # #
number_of_parallel_jobs = 4     # for initial RV guess fits
bf_velocity_span = 800          # km/s


data_template = np.loadtxt(data_path+'5250_40_02_template.dat')
wavelength_template, flux_template = data_template[:, 0], data_template[:, 1]
wavelength, flux_template = spf.resample_to_equal_velocity_steps(wavelength_template, delta_v, flux_template,
                                                                 wavelength_a=5335, wavelength_b=5611)

flux_collection = np.empty((flux_template.size, 13))
bc_rv_cor = np.array([10.1187, 5.1036, 4.3785, -3.9387, -4.6877, -5.6519, -4.9321, -6.1250, -6.3510, -6.5867, -6.8152,
                      -8.6059, -10.0483])
bjdtdb = np.array([56454.7903936, 56477.7589411, 56480.7366352, 56513.6152324, 56516.6366029, 56520.6163267,
                   56517.6346332, 56522.6127962, 56523.5771571, 56524.5905106, 56525.5797603, 56533.5901665,
                   56540.5215877])
for filename in os.listdir(data_path):
    if '15162' in filename:
        print(filename)
        if len(filename) == 16:
            epoch = filename[-5]
        else:
            epoch = filename[-6:-4]
        print(int(epoch))
        data = np.loadtxt(data_path+filename)
        _, flux_collection[:, int(epoch)-1] = spf.resample_to_equal_velocity_steps(data[:, 0], delta_v, data[:, 1],
                                                                                   wavelength)

# Invert fluxes
flux_collection_inverted = 1 - flux_collection
flux_template_inverted   = 1 - flux_template

# # Plot all spectra # #
plt.figure(figsize=(16, 9))
for i in range(0, flux_collection_inverted[0, :].size):
    plt.plot(wavelength, 1-0.05*i -(flux_collection_inverted[:, i]*0.025))
    plt.plot(wavelength, np.ones(shape=wavelength.shape)-0.05*i, '--', color='grey', linewidth=0.5)
plt.xlim([5320, 5620])
plt.xlabel('Wavelength [Å]')
plt.show(block=False)

# # Perform barycentric corrections # #
for i in range(0, flux_collection_inverted[0, :].size):
    flux_collection_inverted[:, i] = ssr.shift_spectrum(flux_collection_inverted[:, i], bc_rv_cor[i]-system_RV_estimate,
                                                        delta_v)

# # Limit data-set to specified area (wavelength_RV_limit) # #
selection_mask = (wavelength > wavelength_RV_limit[0]) & (wavelength < wavelength_RV_limit[1])
wavelength = wavelength[selection_mask]
flux_collection_inverted = flux_collection_inverted[selection_mask, :]
flux_template_inverted = flux_template_inverted[selection_mask]


# # Shorten spectra if uneven # #
if np.mod(wavelength.size, 2) != 0.0:
    wavelength = wavelength[:-1]
    flux_collection_inverted = flux_collection_inverted[:-1, :]
    flux_template_inverted = flux_template_inverted[:-1]


# # Plot all spectra # #
plt.figure(figsize=(16, 9))
for i in range(0, flux_collection_inverted[0, :].size):
    plt.plot(wavelength, 1-0.05*i -(flux_collection_inverted[:, i]*0.025))
    plt.plot(wavelength, np.ones(shape=wavelength.shape)-0.05*i, '--', color='grey', linewidth=0.5)
plt.xlim([5320, 5620])
plt.xlabel('Wavelength [Å]')
plt.show(block=True)


# # Calculate broadening function RVs to use as initial guesses # #
print(np.isnan(flux_collection_inverted).any())
print(np.isnan(wavelength).any())
RV_guesses_A, RV_guesses_B, _ = \
    cRV.radial_velocities_of_multiple_spectra(flux_collection_inverted, flux_template_inverted, delta_v, ifitpar_A,
                                              ifitpar_B, number_of_parallel_jobs, bf_velocity_span, plot=False)
RV_guess_collection = np.empty((RV_guesses_A.size, 2))
RV_guess_collection[:, 0] = RV_guesses_A
if estimate_RVb_from_RVa:
    RV_guesses_B = -RV_guesses_A * 59.49/52.54          # K_B / K_A
RV_guess_collection[:, 1] = RV_guesses_B

 # Separate component spectra and calculate RVs iteratively # #
RV_collection_A, RV_collection_B, separated_flux_A, separated_flux_B = \
    ssr.spectral_separation_routine(flux_collection_inverted, flux_template_inverted, flux_template_inverted,
                                    delta_v, ifitpar_A, ifitpar_B, wavelength, bjdtdb, period=orbital_period_estimate,
                                    bf_velocity_span=bf_velocity_span, RV_guess_collection=RV_guess_collection,
                                    convergence_limit=1E-7)
plt.show(block=True)

# # Plot results # #
plt.figure()
plt.plot(bjdtdb-245000, RV_collection_A, 'r*')
plt.plot(bjdtdb-245000, RV_collection_B, 'b*')
plt.show(block=True)
