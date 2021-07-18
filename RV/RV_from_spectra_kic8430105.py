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
from copy import deepcopy


# # # # Set variables for script # # # #
warnings.filterwarnings("ignore", category=UserWarning)
plt.ion()
data_path = 'Data/unprocessed/NOT/KIC8430105/'
data_out_path = 'Data/processed/NOT/KIC8430105/'

observatory_location = EarthLocation.of_site("lapalma")
observatory_name = "lapalma"
stellar_target = "kic8430105"
wavelength_normalization_limit = (4200, 9600)   # Ångström, limit to data before performing continuum normalization
wavelength_RV_limit = (4700, 5400)              # Ångström, the actual spectrum area used for analysis
wavelength_buffer_size = 50                     # Ångström, padding included at ends of spectra. Useful when doing
                                                # wavelength shifts with np.roll()
wavelength_intervals_error_estimate = 100       # Ångström, size of the intervals used for error estimation on RVs
load_data = True      # Defines if normalized spectrum should be loaded from earlier, or done with AFS_algorithm
plot = False
file_exclude_list = ['FIBl060068_step011_merge.fits']
# use_for_spectral_separation = ['FIBk140069_step011_merge.fits']
use_for_spectral_separation = [
    'FIDi080098_step011_merge.fits',
    'FIBj150080_step011_merge.fits', 'FIDi130112_step011_merge.fits', 'FIBk030043_step011_merge.fits',
    'FIBk060011_step011_merge.fits', 'FIBk140069_step011_merge.fits', 'FIDh160100_step011_merge.fits',
    'FIBi230047_step011_merge.fits', 'FIBi240080_step011_merge.fits', 'FIBI010114_step011_merge.fits',
]
delta_v = 1.0          # interpolation resolution for spectrum in km/s
speed_of_light = scc.c / 1000       # in km/s
estimate_RVb_from_RVa = True        # defines if a guess on RVb should be made in case it cannot be picked up during
                                    # initial fitting
mass_A_estimate = 1.31
mass_B_estimate = 0.83
system_RV_estimate = 16.053  # 16.053 19.44
orbital_period_estimate = 63.33  # only for plotting

# # Stellar parameter estimates (important for limb darkening calculation) # #
Teff_A, Teff_B = 5042, 5621
logg_A, logg_B = 2.78, 4.58
MH_A  , MH_B   = -0.49, -0.49
mTur_A, mTur_B = 2.0, 2.0

# # Initial fit parameters for rotational broadening function fit # #
bf_velocity_span = 300        # broadening function span in velocity space, should be the same for both components
limbd_A = estimate_linear_limbd(wavelength_RV_limit, logg_A, Teff_A, MH_A, mTur_A, loc='Data/tables/atlasco.dat')
limbd_B = estimate_linear_limbd(wavelength_RV_limit, logg_B, Teff_B, MH_B, mTur_B, loc='Data/tables/atlasco.dat')
ifitpar_A = InitialFitParameters(vsini_guess=4.0, spectral_resolution=60000, velocity_fit_width=100, limbd_coef=limbd_A,
                                 smooth_sigma=2.0, bf_velocity_span=bf_velocity_span)
ifitpar_B = InitialFitParameters(vsini_guess=4.0, spectral_resolution=60000, velocity_fit_width=20.0, limbd_coef=limbd_B,
                                 smooth_sigma=4.0, bf_velocity_span=bf_velocity_span)

# # Template Spectra # #
template_spectrum_path_A = 'Data/template_spectra/5000_20_m05p00.ms.fits'
template_spectrum_path_B = 'Data/template_spectra/5500_45_m05p00.ms.fits'

# # Computation parameters # #
number_of_parallel_jobs = 4     # for initial RV guess fits
rv_lower_limit = 10.0           # lower limit for RV_A in order to include a spectrum in the spectral separation
                                # (if lower, components are assumed to be mixed in BF, and left out to not contaminate)
                                # This parameter is only useful when the systemic RV is well-known. Otherwise, set it to
                                # 0.0.
# rv_proximity_limit = 10.0

# # Prepare collection lists and arrays # #
flux_collection_list = []
wavelength_collection_list = []
date_array = np.array([])
RA_array = np.array([])
DEC_array = np.array([])
spectral_separation_array = np.array([])

# # # Load fits files, collect and normalize data # # #
i = 0
for filename in os.listdir(data_path):
    if 'merge.fits' in filename and '.lowSN' not in filename and filename not in file_exclude_list:
        # Load observation
        wavelength, flux, date, ra, dec = spf.load_program_spectrum(data_path+filename)
        date_array = np.append(date_array, date)
        RA_array = np.append(RA_array, ra*15.0)     # converts unit
        DEC_array = np.append(DEC_array, dec)

        # Prepare for continuum fit
        selection_mask = (wavelength > wavelength_normalization_limit[0]) & \
                         (wavelength < wavelength_normalization_limit[1])
        wavelength = wavelength[selection_mask]
        flux = flux[selection_mask]

        # Either load normalized data or do continuum fit
        file_bulk_name = filename[:filename.rfind(".fits")]         # select everything but '.fits'
        if load_data:
            data_in = np.loadtxt('Data/processed/AFS_algorithm/Normalized_Spectrum/'+file_bulk_name+
                                 '_reduced_set.dat')
            wavelength, flux = data_in[:, 0], data_in[:, 1]
        else:
            print('Current file: ', filename)
            wavelength, flux, _ = afs.AFS_merged_spectrum(wavelength, flux, lr_frac=0.2, save_string=file_bulk_name,
                                                          em_line_limit=1.1)

        # Remove values over 1.05 and under 0
        selection_mask = (flux > 1.1) | (flux < 0.0)
        flux = flux[~selection_mask]
        wavelength = wavelength[~selection_mask]

        # Designate if spectrum should be used for spectral separation
        if filename in use_for_spectral_separation:
            spectral_separation_array = np.append(spectral_separation_array, i)

        # Append to collection
        wavelength_collection_list.append(wavelength)
        flux_collection_list.append(flux)
        i += 1

# # Verify RA and DEC # #
RA, DEC = RA_array[0], DEC_array[0]
for i in range(0, len(RA_array)):
    if RA_array[i] == RA and DEC_array[i] == DEC:
        pass
    else:
        warnings.warn("Warning: Either not all RA values equal, or not all DEC values equal")
        print('RA_array:  ', RA_array)
        print('DEC_array: ', DEC_array)

# # # Calculate Barycentric RV Corrections # # #
times = Time(date_array, scale='utc', location=observatory_location)
# Note: this does not correct timezone differences if dates are given in UTC+XX instead of UTC+00. For NOT this is
# probably not an issue since European Western time is UTC+-00. Also, I'm not sure if NOT observations are given in
# UTC or European Western time. # TODO: verify correct time
# Change to Julian Date UTC: probably unnecessary
times.format = 'jd'
times.out_subfmt = 'long'
print()
print("RV correction")
bc_rv_cor, warning, _ = get_BC_vel(times, ra=RA, dec=DEC, starname=stellar_target, ephemeris='de432s',
                                   obsname=observatory_name)
bc_rv_cor = bc_rv_cor/1000      # from m/s to km/s
print(bc_rv_cor)
print(warning)

# # # Calculate JDUTC to BJDTDB correction # # #
print()
print("Time conversion to BJDTDB")
bjdtdb, warning, _ = utc_tdb.JDUTC_to_BJDTDB(times, ra=RA, dec=DEC, starname=stellar_target, obsname=observatory_name)
print(bjdtdb)
print(warning)

# # Plot # #
if plot:
    plt.figure(figsize=(16, 9))
    for i in range(0, len(wavelength_collection_list)):
        # plt.plot(wavelength, 1-0.05*i -(flux_collection_inverted[:, i]*0.025))
        plt.plot(wavelength_collection_list[i], 1 - 0.05 * i - 0.025 + flux_collection_list[i] * 0.025)
        plt.plot(wavelength_collection_list[i], np.ones(shape=wavelength_collection_list[i].shape) - 0.05 * i, '--',
                 color='grey', linewidth=0.7)
    plt.xlim([4600, 6400])
    plt.xlabel('Wavelength [Å]')
    plt.show(block=False)


# # Load template spectrum # #
wavelength_template_A, flux_template_A = spf.load_template_spectrum(template_spectrum_path_A)
flux_template_A = flux_template_A[0, :]     # continuum normalized spectrum only
wavelength_template_B, flux_template_B = spf.load_template_spectrum(template_spectrum_path_B)
flux_template_B = flux_template_B[0, :]

# # Resample to same wavelength grid, equi-spaced in velocity space # #
wavelength, flux_collection_array = spf.resample_to_equal_velocity_steps(wavelength_collection_list, delta_v,
                                                                         flux_collection_list)
_, flux_template_A = spf.resample_to_equal_velocity_steps(wavelength_template_A, delta_v, flux_template_A,
                                                          wavelength_resampled=wavelength)
_, flux_template_B = spf.resample_to_equal_velocity_steps(wavelength_template_B, delta_v, flux_template_B,
                                                          wavelength_resampled=wavelength)

# # Invert fluxes # #
flux_collection_inverted = 1 - flux_collection_array
flux_template_A_inverted = 1 - flux_template_A
flux_template_B_inverted = 1 - flux_template_B


# # Perform barycentric corrections # #
for i in range(0, flux_collection_inverted[0, :].size):
    flux_collection_inverted[:, i] = ssr.shift_spectrum(flux_collection_inverted[:, i],
                                                        bc_rv_cor[i]-system_RV_estimate, delta_v)

# # Shorten spectra if uneven # #
wavelength, [flux_collection_inverted, flux_template_A_inverted, flux_template_B_inverted] = \
    spf.make_spectrum_even(wavelength, [flux_collection_inverted, flux_template_A_inverted, flux_template_B_inverted])


# # Limit data-set to specified area (wavelength_RV_limit) # #
unbuffered_res, buffered_res = spf.limit_wavelength_interval(wavelength_RV_limit, wavelength, flux_collection_inverted,
                                                             buffer_size=wavelength_buffer_size, even_length=True)
(wavelength_new, flux_collection_inverted) = unbuffered_res
(wavelength_buffered, flux_collection_inverted_buffered, buffer_mask_new, buffer_mask) = buffered_res
# buffer_mask_new is for use with wavelength_buffered and flux_coll... buffer_mask is for use with input data

unbuffered_res, buffered_res = spf.limit_wavelength_interval(wavelength_RV_limit, wavelength, flux_template_A_inverted,
                                                             buffer_mask=buffer_mask, even_length=True)
flux_template_A_inverted = unbuffered_res[1]
flux_template_A_inverted_buffered = buffered_res[1]

unbuffered_res, buffered_res = spf.limit_wavelength_interval(wavelength_RV_limit, wavelength, flux_template_B_inverted,
                                                             buffer_mask=buffer_mask, even_length=True)
flux_template_B_inverted = unbuffered_res[1]
flux_template_B_inverted_buffered = buffered_res[1]

wavelength = wavelength_new
buffer_mask = buffer_mask_new


# # Plot all spectra # #
if plot:
    plt.figure(figsize=(16, 9))
    for i in range(0, flux_collection_inverted[0, :].size):
        plt.plot(wavelength_buffered, 1 - 0.05 * i - (flux_collection_inverted_buffered[:, i] * 0.025), '--',
                 color='grey',
                 linewidth=0.3)
        plt.plot(wavelength, 1 - 0.05 * i - (flux_collection_inverted[:, i] * 0.025))
        plt.plot(wavelength, np.ones(shape=wavelength.shape) - 0.05 * i, '--', color='grey', linewidth=0.5)
    plt.xlim([4600, 6400])
    plt.xlabel('Wavelength [Å]')
    plt.show(block=True)


# # Calculate broadening function RVs to use as initial guesses # #
RV_guesses_A, RV_guesses_B, _ = \
    cRV.radial_velocities_of_multiple_spectra(flux_collection_inverted, flux_template_A_inverted, delta_v, ifitpar_A,
                                              ifitpar_B, number_of_parallel_jobs, plot=False)
RV_guess_collection = np.empty((RV_guesses_A.size, 2))
RV_guess_collection[:, 0] = RV_guesses_A
if estimate_RVb_from_RVa:
    RV_guesses_B = -RV_guesses_A * (mass_A_estimate/mass_B_estimate)

RV_guess_collection[:, 1] = RV_guesses_B


# # Separate component spectra and calculate RVs iteratively # #
RV_collection_A, RV_collection_B, separated_flux_A, separated_flux_B, wavelength, iteration_errors, ifitpars = \
    ssr.spectral_separation_routine(
        flux_collection_inverted_buffered, flux_template_A_inverted_buffered, flux_template_B_inverted_buffered,
        delta_v, ifitpar_A, ifitpar_B, wavelength_buffered,  bjdtdb, period=orbital_period_estimate, iteration_limit=7,
        RV_guess_collection=RV_guess_collection, convergence_limit=1E-1, buffer_mask=buffer_mask,
        rv_lower_limit=rv_lower_limit, suppress_print='scs', plot=False, estimate_error=False, return_unbuffered=False,
        use_spectra=spectral_separation_array
    )
# plt.show(block=True)

# plt.figure()
bad_data_mask = np.abs(RV_collection_A) < rv_lower_limit
phase_B = np.mod(bjdtdb[~bad_data_mask], orbital_period_estimate)/orbital_period_estimate
phase_A = np.mod(bjdtdb, orbital_period_estimate) / orbital_period_estimate
# plt.plot(phase_B, RV_collection_B[~bad_data_mask], 'rp', markersize=3)
# plt.plot(phase_A, RV_collection_A, 'bp', markersize=3)

# RV_errors_A = iteration_errors[0]
# RV_errors_B = iteration_errors[1]

# print('RV errors lower limit')
# print(RV_errors_A)
# print(RV_errors_B)

# # Calculate uncertainties by splitting up into smaller intervals # #
wiee = wavelength_intervals_error_estimate
(RV_errors_A, RV_errors_B), (RV_estimates_A, RV_estimates_B) = ssr.estimate_errors(
    wiee, flux_collection_inverted_buffered, flux_template_A_inverted_buffered, flux_template_B_inverted_buffered,
    separated_flux_A, separated_flux_B, delta_v, ifitpars[0], ifitpars[1], wavelength, RV_collection_A, RV_collection_B,
    bjdtdb, wavelength_buffer_size, plot=plot, period=orbital_period_estimate
)


# bad_data_mask = np.abs(RV_collection_A-RV_collection_B) < rv_proximity_limit
bad_data_mask = np.abs(RV_collection_A) < rv_lower_limit
bjdtdb_B = bjdtdb[~bad_data_mask]
# RV_errors_B = RV_errors_B[~bad_data_mask]
# RV_collection_B = RV_collection_B[~bad_data_mask]


# plt.errorbar(np.mod(bjdtdb, orbital_period_estimate)/orbital_period_estimate, RV_collection_A, yerr=RV_errors_A,
#              fmt='b*')
# plt.errorbar(np.mod(bjdtdb_B, orbital_period_estimate)/orbital_period_estimate, RV_collection_B, yerr=RV_errors_B,
#              fmt='r*')
# plt.xlabel('Orbital Phase')
# plt.ylabel('Radial Velocity (km/s)')
# plt.show(block=True)


# # Save result # #
save_data = np.empty((RV_collection_A.size, 3))
save_data[:, 0] = bjdtdb
save_data[:, 1] = RV_collection_A
save_data[:, 2] = RV_errors_A
np.savetxt('Data/processed/RV_results/rvA_not_8430105_4700_5400_100_3.txt', save_data)

save_data = np.empty((RV_collection_B[~bad_data_mask].size, 3))
save_data[:, 0] = bjdtdb_B
save_data[:, 1] = RV_collection_B[~bad_data_mask]
save_data[:, 2] = RV_errors_B[~bad_data_mask]
np.savetxt('Data/processed/RV_results/rvB_not_8430105_4700_5400_100_3.txt', save_data)

RV_coll = np.empty(shape=(RV_collection_A.size, 2))
RV_coll[:, 0] = deepcopy(RV_collection_A)
RV_coll[:, 1] = deepcopy(RV_collection_B)
RV_errors_A2, RV_errors_B2 = ssr.estimate_errors_2(
    wiee, flux_collection_inverted_buffered, flux_template_A_inverted_buffered, flux_template_B_inverted_buffered,
    delta_v, ifitpar_A, ifitpar_B, wavelength_buffered, bjdtdb, RV_coll, convergence_limit=1E-1, plot=False,
    period=orbital_period_estimate, wavelength_buffer_size=wavelength_buffer_size, rv_lower_limit=rv_lower_limit,
    suppress_print='all', use_spectra=spectral_separation_array)

# # Save result # #
save_data = np.empty((RV_collection_A.size, 3))
save_data[:, 0] = bjdtdb
save_data[:, 1] = RV_collection_A
save_data[:, 2] = RV_errors_A2
np.savetxt('Data/processed/RV_results/rvA_not_8430105_4700_5400_100_3errors2.txt', save_data)

save_data = np.empty((RV_collection_B[~bad_data_mask].size, 3))
save_data[:, 0] = bjdtdb_B
save_data[:, 1] = RV_collection_B[~bad_data_mask]
save_data[:, 2] = RV_errors_B2[~bad_data_mask]
np.savetxt('Data/processed/RV_results/rvB_not_8430105_4700_5400_100_3errors2.txt', save_data)
