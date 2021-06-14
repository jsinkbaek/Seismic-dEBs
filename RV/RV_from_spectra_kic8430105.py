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


# # # # Set variables for script # # # #
# matplotlib.use('Qt5Agg')
plt.ion()
data_path = 'Data/unprocessed/NOT/KIC8430105/'
data_out_path = 'Data/processed/NOT/KIC8430105/'

observatory_location = EarthLocation.of_site("lapalma")
observatory_name = "lapalma"
stellar_target = "kic8430105"
wavelength_normalization_limit = (4200, 9600)
wavelength_RV_limit = (5200, 5700)
load_data = True      # Defines if normalized spectrum should be loaded from earlier, or done with AFS_algorithm
#  afs_exclude_list = ['FIBj010048_step011_merge.fits', 'FIDi130112_step011_merge.fits', 'FIDh160100_step011_merge.fits', 'FIBi240080_step011_merge.fits', 'FIBi300038_step011_merge.fits', 'FIBi230047_step011_merge.fits', 'FIBk030043_step011_merge.fits', 'FIDi080098_step011_merge.fits', 'FIBj030100_step011_merge.fits', 'FIBk050063_step011_merge.fits', 'FIBl060068_step011_merge.fits', 'FIBj150080_step011_merge.fits', 'FIDi090065_step011_merge.fits', 'FIBj040099_step011_merge.fits', 'FIBl010114_step011_merge.fits', 'FIBk060011_step011_merge.fits', 'FIBk140069_step011_merge.fits', 'FIBi290054_step011_merge.fits', 'FIBk230070_step011_merge.fits']
afs_exclude_list = ['FIBl060068_step011_merge.fits']
delta_v = 1.0          # interpolation resolution for spectrum in km/s
speed_of_light = scc.c / 1000    # in km/s
estimate_RVb_from_RVa = True        # defines if a guess on RVb should be made in case it cannot be picked up during
                                    # initial fitting
mass_A_estimate = 1.31
mass_B_estimate = 0.83
system_RV_estimate = 16.053
orbital_period_estimate = 63.33  # only for plotting

# # Stellar parameter estimates (important for limb darkening calculation) # #
Teff_A, Teff_B = 5042, 5621
logg_A, logg_B = 2.78, 4.58
MH_A  , MH_B   = -0.49, -0.49
mTur_A, mTur_B = 2.0, 2.0

# # Initial fit parameters for rotational broadening function fit # #
limbd_A = estimate_linear_limbd(wavelength_RV_limit, logg_A, Teff_A, MH_A, mTur_A, loc='Data/tables/atlasco.dat')
limbd_B = estimate_linear_limbd(wavelength_RV_limit, logg_B, Teff_B, MH_B, mTur_B, loc='Data/tables/atlasco.dat')
ifitpar_A = InitialFitParameters(vsini_guess=4.0, spectral_resolution=60000, velocity_fit_width=100, limbd_coef=limbd_A,
                                 smooth_sigma=3.0)
ifitpar_B = InitialFitParameters(vsini_guess=4.0, spectral_resolution=60000, velocity_fit_width=25, limbd_coef=limbd_B,
                                 smooth_sigma=4.0)

# # Template Spectra # #
template_spectrum_path_A = 'Data/template_spectra/5000_20_m05p00.ms.fits'
template_spectrum_path_B = 'Data/template_spectra/5500_45_m05p00.ms.fits'

# # Broadening function and radial velocity parameters # #
number_of_parallel_jobs = 4     # for initial RV guess fits
bf_velocity_span = 250          # km/s


# # Prepare collection lists and arrays # #
flux_collection_list = []
wavelength_collection_list = []
date_array = np.array([])
RA_array = np.array([])
DEC_array = np.array([])

# # # Load fits files, collect and normalize data # # #
for filename in os.listdir(data_path):
    if 'merge.fits' in filename and '.lowSN' not in filename and filename not in afs_exclude_list:
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

        # Append to collection
        wavelength_collection_list.append(wavelength)
        flux_collection_list.append(flux)

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
plt.figure(figsize=(16, 9))
for i in range(0, len(wavelength_collection_list)):
    # plt.plot(wavelength, 1-0.05*i -(flux_collection_inverted[:, i]*0.025))
    plt.plot(wavelength_collection_list[i], 1-0.05*i - 0.025 + flux_collection_list[i]*0.025)
    plt.plot(wavelength_collection_list[i], np.ones(shape=wavelength_collection_list[i].shape)-0.05*i, '--',
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


# # Limit data-set to specified area (wavelength_RV_limit) # #
selection_mask = (wavelength > wavelength_RV_limit[0]) & (wavelength < wavelength_RV_limit[1])
wavelength = wavelength[selection_mask]
flux_collection_inverted = flux_collection_inverted[selection_mask, :]
flux_template_A_inverted = flux_template_A_inverted[selection_mask]
flux_template_B_inverted = flux_template_B_inverted[selection_mask]


# # Shorten spectra if uneven # #
if np.mod(wavelength.size, 2) != 0.0:
    wavelength = wavelength[:-1]
    flux_collection_inverted = flux_collection_inverted[:-1, :]
    flux_template_A_inverted = flux_template_A_inverted[:-1]
    flux_template_B_inverted = flux_template_B_inverted[:-1]

# # Plot all spectra # #
plt.figure(figsize=(16, 9))
for i in range(0, flux_collection_inverted[0, :].size):
    plt.plot(wavelength, 1-0.05*i -(flux_collection_inverted[:, i]*0.025))
    plt.plot(wavelength, np.ones(shape=wavelength.shape)-0.05*i, '--', color='grey', linewidth=0.5)
plt.xlim([4600, 6400])
plt.xlabel('Wavelength [Å]')
plt.show(block=True)

# # Calculate broadening function RVs to use as initial guesses # #
RV_guesses_A, RV_guesses_B, _ = \
    cRV.radial_velocities_of_multiple_spectra(flux_collection_inverted, flux_template_A_inverted, delta_v, ifitpar_A,
                                              ifitpar_B, number_of_parallel_jobs, bf_velocity_span, plot=False)
RV_guess_collection = np.empty((RV_guesses_A.size, 2))
RV_guess_collection[:, 0] = RV_guesses_A
if estimate_RVb_from_RVa:
    RV_guesses_B = -RV_guesses_A * (mass_A_estimate/mass_B_estimate)

RV_guess_collection[:, 1] = RV_guesses_B


# # Separate component spectra and calculate RVs iteratively # #
RV_collection_A, RV_collection_B, separated_flux_A, separated_flux_B = \
    ssr.spectral_separation_routine(flux_collection_inverted, flux_template_A_inverted, flux_template_B_inverted,
                                    delta_v, ifitpar_A, ifitpar_B, wavelength, bjdtdb, period=orbital_period_estimate,
                                    bf_velocity_span=bf_velocity_span, RV_guess_collection=RV_guess_collection,
                                    convergence_limit=1E-7)
plt.show(block=True)

# # Plot results # #
plt.figure()
plt.plot(bjdtdb-245000, RV_collection_A, 'r*')
plt.plot(bjdtdb-245000, RV_collection_B, 'b*')
plt.show(block=True)

