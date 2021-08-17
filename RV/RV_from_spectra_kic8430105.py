import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import os
from barycorrpy import get_BC_vel, utc_tdb
import RV.library.spectrum_processing_functions as spf
import warnings
import scipy.constants as scc
from RV.library.initial_fit_parameters import InitialFitParameters
import RV.library.spectral_separation_routine as ssr
from RV.library.linear_limbd_coeff_estimate import estimate_linear_limbd
import RV.library.broadening_function_svd as bfsvd
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
import RV.library.calculate_radial_velocities as cRV

matplotlib.rcParams.update({'font.size': 25})

# # # # Set variables for script # # # #
warnings.filterwarnings("ignore", category=UserWarning)
plt.ion()
data_path = 'Data/unprocessed/NOT/KIC8430105/'
data_out_path = 'Data/processed/NOT/KIC8430105/'

observatory_location = EarthLocation.of_site("lapalma")
observatory_name = "lapalma"
stellar_target = "kic8430105"
wavelength_normalization_limit = (4450, 7000)   # Ångström, limit to data before performing continuum normalization
wavelength_RV_limit = (5000, 5600)              # Ångström, the actual spectrum area used for analysis
wavelength_buffer_size = 25                     # Ångström, padding included at ends of spectra. Useful when doing
                                                # wavelength shifts with np.roll()
wavelength_intervals_error_estimate = 150       # Ångström, size of the intervals used for error estimation on RVs
load_data = True      # Defines if normalized spectrum should be loaded from earlier, or done with AFS_algorithm
plot = False
file_exclude_list = []  # ['FIBl060068_step011_merge.fits']
# use_for_spectral_separation_A = [
#    'FIBj030100_step011_merge.fits', 'FIBj030108_step011_merge.fits',
#    'FIBj040099_step011_merge.fits'
# ]
use_for_spectral_separation_A = [
    'FIDi080098_step011_merge.fits', 'FIDi090065_step011_merge.fits',
    'FIBj150080_step011_merge.fits', 'FIDi130112_step011_merge.fits', 'FIBk030043_step011_merge.fits',
    'FIBk050063_step011_merge.fits',
    'FIBk140069_step011_merge.fits', 'FIDh160100_step011_merge.fits',
    'FIBi230047_step011_merge.fits', 'FIBi240080_step011_merge.fits'
    ]
# not used: 'FIBk230070_step011_merge.fits', 'FIBk060011_step011_merge.fits',
use_for_spectral_separation_B = [
    'FIDi080098_step011_merge.fits', 'FIDi090065_step011_merge.fits',
    'FIBj150080_step011_merge.fits', 'FIDi130112_step011_merge.fits', 'FIBk030043_step011_merge.fits',
    'FIBk050063_step011_merge.fits',
    'FIBk140069_step011_merge.fits', 'FIDh160100_step011_merge.fits',
    'FIBi230047_step011_merge.fits', 'FIBi240080_step011_merge.fits'
    ]
delta_v = 1.0          # interpolation resolution for spectrum in km/s
speed_of_light = scc.c / 1000       # in km/s
estimate_RVb_from_RVa = True        # defines if a guess on RVb should be made in case it cannot be picked up during
                                    # initial fitting
mass_A_estimate = 1.31
mass_B_estimate = 0.83
system_RV_estimate = 16.053  # 16.053 19.44
orbital_period_estimate = 63.33  # only for plotting

# # Stellar parameter estimates (relevant for limb darkening calculation) # #
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
                                 smooth_sigma=4.0, bf_velocity_span=bf_velocity_span, ignore_at_phase=(0.98, 0.02))

# # Template Spectra # #
template_spectrum_path_A = 'Data/template_spectra/5000_20_m05p00.ms.fits'
template_spectrum_path_B = 'Data/template_spectra/5500_45_m05p00.ms.fits'

# # Computation parameters # #
number_of_parallel_jobs = 4     # for initial RV guess fits
rv_lower_limit = 12             # lower limit for RV_A in order to include a spectrum in the spectral separation
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
spectral_separation_array_A = np.array([])
spectral_separation_array_B = np.array([])

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

        # Remove values under 0
        selection_mask = (flux >= 0.0)
        flux = flux[selection_mask]
        wavelength = wavelength[selection_mask]

        # Performs continuum fit and reduces emission lines (by removing above 2.5 std from fitted continuum)
        wavelength, flux = spf.simple_normalizer(wavelength, flux, reduce_em_lines=True, plot=False)

        # Designate if spectrum should be used for spectral separation
        if filename in use_for_spectral_separation_A:
            spectral_separation_array_A = np.append(spectral_separation_array_A, i)
        if filename in use_for_spectral_separation_B:
            spectral_separation_array_B = np.append(spectral_separation_array_B, i)

        # Append to collection
        wavelength_collection_list.append(wavelength)
        flux_collection_list.append(flux)
        i += 1


ifitpar_A.use_for_spectral_separation = spectral_separation_array_A
ifitpar_B.use_for_spectral_separation = spectral_separation_array_B


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
wavelength, (flux_collection_array, flux_template_A, flux_template_B) = spf.resample_multiple_spectra(
    delta_v, (wavelength_collection_list, flux_collection_list), (wavelength_template_A, flux_template_A),
    (wavelength_template_B, flux_template_B)
)

# # Invert fluxes # #
flux_collection_inverted = 1 - flux_collection_array
flux_template_A_inverted = 1 - flux_template_A
flux_template_B_inverted = 1 - flux_template_B

# # Perform barycentric corrections # #
for i in range(0, flux_collection_inverted[0, :].size):
    flux_collection_inverted[:, i] = ssr.shift_spectrum(
        flux_collection_inverted[:, i], bc_rv_cor[i]-system_RV_estimate, delta_v
    )

# # Shorten spectra if uneven # #
wavelength, [flux_collection_inverted, flux_template_A_inverted, flux_template_B_inverted] = \
    spf.make_spectrum_even(wavelength, [flux_collection_inverted, flux_template_A_inverted, flux_template_B_inverted])


# # Limit data-set to specified area (wavelength_RV_limit) # #
wavelength, flux_unbuffered_list, wavelength_buffered, flux_buffered_list, buffer_mask = \
    spf.limit_wavelength_interval_multiple_spectra(
        wavelength_RV_limit, wavelength, flux_collection_inverted, flux_template_A_inverted, flux_template_B_inverted,
        buffer_size=wavelength_buffer_size, even_length=True
    )
[flux_collection_inverted, flux_template_A_inverted, flux_template_B_inverted] = flux_unbuffered_list
[flux_collection_inverted_buffered, flux_template_A_inverted_buffered, flux_template_B_inverted_buffered] = \
    flux_buffered_list


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

    # # Calculate one cross-correlation (plotting reasons) # #
    temp_flux = 1 - flux_collection_inverted[:, 8]
    temp_flux = np.mean(temp_flux) - temp_flux
    corr = np.correlate(temp_flux, flux_template_A_inverted, mode='same')
    corr = corr / np.max(corr)
    velocity_shifts_corr = np.linspace(-corr.size // 2 * delta_v, corr.size // 2 * delta_v, corr.size)

    plt.figure(figsize=(16, 9))
    plt.plot(velocity_shifts_corr, corr, linewidth=3)
    plt.xlabel('Velocity Shift [km/s]')
    plt.ylabel('Normalized Cross-Correlation')
    plt.xlim([-100, 100])
    plt.tight_layout()
    plt.savefig(fname='../figures/report/RV/cross_correlation.png', dpi=400)
    plt.close()

    # # Calculate broadening function for spectrum 8 (plotting reasons) # #
    BF_temp = bfsvd.BroadeningFunction(flux_collection_inverted[:, 8], flux_template_A_inverted, bf_velocity_span,
                                       delta_v)
    BF_temp.smooth_sigma = ifitpar_A.bf_smooth_sigma
    BF_temp.solve()
    BF_temp.smooth()

    plt.figure(figsize=(16, 9))
    plt.plot(BF_temp.velocity, BF_temp.bf_smooth / np.max(BF_temp.bf_smooth), 'r', linewidth=3)
    plt.plot(velocity_shifts_corr, corr, 'b', linewidth=3)
    plt.xlabel('Velocity Shift [km/s]')
    plt.legend(['Smoothed Broadening Function', 'Cross-Correlation'])
    plt.xlim([-100, 100])
    plt.tight_layout()
    plt.savefig(fname='../figures/report/RV/bf_cc.png', dpi=400)
    plt.close()


# # Calculate broadening function RVs to use as initial guesses # #
RV_guesses_A, _ = cRV.radial_velocities_of_multiple_spectra(
    flux_collection_inverted, flux_template_A_inverted, delta_v, ifitpar_A, number_of_parallel_jobs=4,
    plot=False
)
RV_guess_collection = np.empty((RV_guesses_A.size, 2))
RV_guess_collection[:, 0] = RV_guesses_A
RV_guesses_B = -RV_guesses_A * (mass_A_estimate / mass_B_estimate)

RV_guess_collection[:, 1] = RV_guesses_B

# # Separate component spectra and calculate RVs iteratively # #
RV_collection_A, RV_collection_B, separated_flux_A, separated_flux_B, wavelength, ifitpars, RVb_flags = \
    ssr.spectral_separation_routine(
        flux_collection_inverted_buffered, flux_template_A_inverted_buffered, flux_template_B_inverted_buffered,
        delta_v, ifitpar_A, ifitpar_B, wavelength_buffered, bjdtdb - (2400000 + 54976.6348),
        period=orbital_period_estimate,
        iteration_limit=2, RV_guess_collection=RV_guess_collection, convergence_limit=1E-2, buffer_mask=buffer_mask,
        rv_lower_limit=rv_lower_limit, suppress_print='scs', plot=False, return_unbuffered=False,
        ignore_component_B=False
    )
# plt.show(block=True)
plt.close('all')
"""
# plt.figure()
bad_data_mask = np.abs(RV_collection_A) < rv_lower_limit
phase_B = np.mod(bjdtdb[~bad_data_mask], orbital_period_estimate) / orbital_period_estimate
phase_A = np.mod(bjdtdb, orbital_period_estimate) / orbital_period_estimate

# # Calculate uncertainties by splitting up into smaller intervals # #
wiee = wavelength_intervals_error_estimate
(RV_errors_A, RV_errors_B), (RV_estimates_A, RV_estimates_B) = ssr.estimate_errors_rv_only(
    wiee, flux_collection_inverted_buffered, flux_template_A_inverted_buffered, flux_template_B_inverted_buffered,
    separated_flux_A, separated_flux_B, delta_v, ifitpars[0], ifitpars[1], wavelength, RV_collection_A,
    RV_collection_B, bjdtdb, wavelength_buffer_size, plot=plot, period=orbital_period_estimate
    )

# bad_data_mask = np.abs(RV_collection_A-RV_collection_B) < rv_proximity_limit
bad_data_mask = np.abs(RV_collection_A) < rv_lower_limit
bjdtdb_B = bjdtdb[~bad_data_mask]

# # Save result # #

save_data = np.empty((RV_collection_A.size, 3))
save_data[:, 0] = bjdtdb
save_data[:, 1] = RV_collection_A + system_RV_estimate
save_data[:, 2] = RV_errors_A
np.savetxt('Data/processed/RV_results/rvA_not_8430105_4700_5400_100.txt', save_data)

save_data = np.empty((RV_collection_B[~bad_data_mask].size, 3))
save_data[:, 0] = bjdtdb_B
save_data[:, 1] = RV_collection_B[~bad_data_mask] + system_RV_estimate
save_data[:, 2] = RV_errors_B[~bad_data_mask]
np.savetxt('Data/processed/RV_results/rvB_not_8430105_4700_5400_100.txt', save_data)


_, RV_A_from_previous, _ = np.loadtxt('Data/processed/RV_results/rvA_not_8430105_4700_5400_100.txt',
                                      unpack=True)
RV_A_from_previous = RV_A_from_previous - system_RV_estimate

bad_data_mask = np.abs(RV_A_from_previous) < rv_lower_limit
bjdtdb_B = bjdtdb[~bad_data_mask]

if estimate_RVb_from_RVa:
    RV_guesses_B = -RV_A_from_previous * (mass_A_estimate/mass_B_estimate)

RV_coll = np.empty(shape=(RV_A_from_previous.size, 2))
RV_coll[:, 0] = deepcopy(RV_A_from_previous)
RV_coll[:, 1] = deepcopy(RV_guesses_B)
wiee = wavelength_intervals_error_estimate

RV_errors_A2, RV_errors_B2, (RV_A_individual, RV_B_individual) = ssr.estimate_errors_2(
    wiee, flux_collection_inverted_buffered, flux_template_A_inverted_buffered, flux_template_B_inverted_buffered,
    delta_v, ifitpar_A, ifitpar_B, wavelength_buffered, bjdtdb - (2400000 + 54976.6348), RV_coll,
    convergence_limit=5E-2, plot=True,
    period=orbital_period_estimate, wavelength_buffer_size=wavelength_buffer_size, rv_lower_limit=rv_lower_limit,
    suppress_print='all', save_bf_plots=True, iteration_limit=5
)


# # Save result # #
save_data = np.empty((RV_collection_A.size, 3))
save_data[:, 0] = bjdtdb
save_data[:, 1] = RV_collection_A + system_RV_estimate
save_data[:, 2] = RV_errors_A2
np.savetxt('Data/processed/RV_results/rvA_not_8430105_4500_6900_100_errors2.txt', save_data)

save_data = np.empty((RV_collection_B[~bad_data_mask].size, 3))
save_data[:, 0] = bjdtdb_B
save_data[:, 1] = RV_collection_B[~bad_data_mask] + system_RV_estimate
save_data[:, 2] = RV_errors_B2[~bad_data_mask]
np.savetxt('Data/processed/RV_results/rvB_not_8430105_4500_6900_100_errors2.txt', save_data)


# # Save individual interval RV measurements
save_data = np.empty((RV_A_from_previous.size, 2+RV_A_individual[0, :].size))
save_data[:, 0] = bjdtdb
save_data[:, 1] = RV_errors_A2
save_data[:, 2:] = RV_A_individual + system_RV_estimate
np.savetxt('Data/processed/RV_results/interval_results/rvA_not_8430105_4500_6900_150_intervals.txt', save_data)

save_data = np.empty((RV_guesses_B[~bad_data_mask].size, 2+RV_B_individual[0, :].size))
save_data[:, 0] = bjdtdb_B
save_data[:, 1] = RV_errors_B2[~bad_data_mask]
save_data[:, 2:] = RV_B_individual[~bad_data_mask, :] + system_RV_estimate
np.savetxt('Data/processed/RV_results/interval_results/rvB_not_8430105_4500_6900_150_intervals.txt', save_data)
"""
