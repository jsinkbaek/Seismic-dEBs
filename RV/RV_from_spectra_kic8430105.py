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

# # Prepare collector lists and arrays # #
flux_collector_list = []
wavelength_collector_list = []
date_array = np.array([])
RA_array = np.array([])
DEC_array = np.array([])

# # # Load fits files, collect and normalize data # # #
for filename in os.listdir(data_path):
    if 'merge.fits' in filename and '.lowSN' not in filename:
        with fits.open(data_path + filename) as hdul:
            hdr = hdul[0].header
            flux = hdul[0].data
            wl0, delta_wl = hdr['CRVAL1'], hdr['CDELT1']
            date_array = np.append(date_array, hdr['DATE-AVG'])
            RA_array = np.append(RA_array, hdr['OBJRA']*15.0)       # convert unit
            DEC_array = np.append(DEC_array, hdr['OBJDEC'])
        # Prepare for continuum fit #
        wavelength = np.linspace(wl0, wl0 + delta_wl * flux.size, flux.size)
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
            wavelength, flux, _ = spf.AFS_algorithm(wavelength, flux, lr_frac=0.2, save_string=file_bulk_name)

        # Limit normalized data set to smaller wavelength range for RV analysis, and append to collector
        selection_mask = (wavelength > wavelength_RV_limit[0]) & (wavelength < wavelength_RV_limit[1])
        wavelength_collector_list = np.append(wavelength_collector_list, wavelength[selection_mask])
        flux_collector_list = np.append(flux_collector_list, flux[selection_mask])

wavelength, flux_collector_array = spf.resample_to_equal_velocity_steps(wavelength_collector_list, flux_collector_list,
                                                                        delta_v)

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
# UTC or European Western time.
# Change to Julian Date UTC:
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
