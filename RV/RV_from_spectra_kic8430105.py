import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation
import os
from scipy.interpolate import interp1d
from barycorrpy import get_BC_vel, utc_tdb
import RV.library.spectrum_processing_functions as spf
import warnings
import scipy.constants as scc
import RV.library.AFS_algorithm as afs
import RV.library.calculate_radial_velocities as cRV
from RV.library.initial_fit_parameters import InitialFitParameters
import RV.library.spectral_separation_routine as ssr
from RV.library.linear_limbd_coeff_estimate import estimate_linear_limbd


# # # # Set variables for script # # # #
data_path = 'RV/Data/unprocessed/NOT/KIC8430105/'
data_out_path = 'RV/Data/processed/NOT/KIC8430105/'

observatory_location = EarthLocation.of_site("lapalma")
observatory_name = "lapalma"
stellar_target = "kic8430105"
wavelength_normalization_limit = (4200, 9600)
wavelength_RV_limit = (5300, 5700)
load_data = True       # Defines if normalized spectrum should be loaded from earlier, or done with AFS_algorithm
delta_v = 1.0          # interpolation resolution for spectrum in km/s
speed_of_light = scc.c / 1000    # in km/s

# # Stellar parameter estimates (important for limbd) # #
Teff_A, Teff_B = 5042, 5621
logg_A, logg_B = 2.78, 4.58
MH_A  , MH_B   = -0.49, -0.49
mTur_A, mTur_B = 2.0, 2.0

# # Initial fit parameters for rotational broadening function fit # #
limbd_A = estimate_linear_limbd(wavelength_RV_limit, logg_A, Teff_A, MH_A, mTur_A)
limbd_B = estimate_linear_limbd(wavelength_RV_limit, logg_B, Teff_B, MH_B, mTur_B)
ifitpar_A = InitialFitParameters(vsini_guess=1.0, spectral_resolution=60000, velocity_fit_width=200, limbd_coef=limbd_A)
ifitpar_B = InitialFitParameters(vsini_guess=1.0, spectral_resolution=60000, velocity_fit_width=350, limbd_coef=limbd_B)

# # Template Spectra # #
template_spectrum_path_A = '../Data/template_spectra/5000_20_m05p00.ms.fits'
template_spectrum_path_B = '../Data/template_spectra/5500_20_m05p00.ms.fits'

# # Broadening function and radial velocity parameters # #
bf_smooth_sigma = 4.0
number_of_parallel_jobs = 8     # for initial RV guess fits
bf_velocity_span = 381

# # Prepare collection lists and arrays # #
flux_collection_list = []
wavelength_collection_list = []
date_array = np.array([])
RA_array = np.array([])
DEC_array = np.array([])

# # # Load fits files, collect and normalize data # # #
for filename in os.listdir(data_path):
    if 'merge.fits' in filename and '.lowSN' not in filename:
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
            data_in = np.loadtxt('RV/Data/processed/AFS_algorithm/Normalized_Spectrum/'+file_bulk_name+
                                 '_reduced_set.dat')
            wavelength, flux = data_in[:, 0], data_in[:, 1]
        else:
            wavelength, flux, _ = afs.AFS_merged_spectrum(wavelength, flux, lr_frac=0.2, save_string=file_bulk_name)

        # Limit normalized data set to smaller wavelength range for RV analysis, and append to collection
        selection_mask = (wavelength > wavelength_RV_limit[0]) & (wavelength < wavelength_RV_limit[1])
        wavelength_collection_list = wavelength_collection_list.append(wavelength[selection_mask])
        flux_collection_list = flux_collection_list.append(flux[selection_mask])

# # Load template spectrum # #
wavelength_template_A, flux_template_A = spf.load_template_spectrum(template_spectrum_path_A)
flux_template_A = flux_template_A[0, :]     # continuum normalized spectrum only
wavelength_template_B, flux_template_B = spf.load_template_spectrum(template_spectrum_path_B)
flux_template_B = flux_template_B[0, :]

# # Resample to same wavelength grid, equi-spaced in velocity space # #
wavelength, flux_collection_array = spf.resample_to_equal_velocity_steps(wavelength_collection_list,
                                                                         flux_collection_list,
                                                                         delta_v)
_, flux_template_A = spf.resample_to_equal_velocity_steps(wavelength_template_A, delta_v, flux_template_A,
                                                          wavelength_resampled=wavelength)
_, flux_template_B = spf.resample_to_equal_velocity_steps(wavelength_template_B, delta_v, flux_template_B,
                                                          wavelength_resampled=wavelength)

# # Invert fluxes # #
flux_collection_inverted = 1 - flux_collection_array
flux_template_A_inverted = 1 - flux_template_A
flux_template_B_inverted = 1 - flux_template_B

# # Calculate broadening function RVs to use as initial guesses # #
RV_guesses_A, RV_guesses_B, _ = \
    cRV.radial_velocities_of_multiple_spectra(flux_collection_inverted, flux_template_A_inverted, delta_v, ifitpar_A,
                                              ifitpar_B, bf_smooth_sigma, number_of_parallel_jobs, bf_velocity_span,
                                              plot=True)
RV_guess_collection = np.empty((RV_guesses_A.size, 2))
RV_guess_collection[:, 0] = RV_guesses_A
RV_guess_collection[:, 1] = RV_guesses_B

# # Separate component spectra and calculate RVs iteratively # #
RV_collection_A, RV_collection_B, separated_flux_A, separated_flux_B = \
    ssr.spectral_separation_routine(flux_collection_inverted, flux_template_A_inverted, flux_template_B_inverted,
                                    delta_v, ifitpar_A, ifitpar_B, bf_smooth_sigma,  bf_velocity_span=bf_velocity_span,
                                    RV_guess_collection=RV_guess_collection)


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

print(bc_rv_cor)
print(warning)

# # # Calculate JDUTC to BJDTDB correction # # #
print()
print("Time conversion to BJDTDB")
bjdtdb, warning, _ = utc_tdb.JDUTC_to_BJDTDB(times, ra=RA, dec=DEC, starname=stellar_target, obsname=observatory_name)
print(bjdtdb)
print(warning)
