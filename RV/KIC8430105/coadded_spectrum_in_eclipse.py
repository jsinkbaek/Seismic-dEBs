import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation
import os
from barycorrpy import get_BC_vel, utc_tdb
import RV.library.spectrum_processing_functions as spf
import matplotlib.pyplot as plt
import matplotlib
import scipy.constants as scc
import RV.library.spectral_separation_routine as ssr


matplotlib.rcParams.update({'font.size': 25})

observatory = 'lapalma'
observatory_location = EarthLocation.of_site(observatory)

stellar_target = "kic8430105"
wavelength_normalization_limit = (4450, 7000)   # Ångström, limit to data before performing continuum normalization

delta_v = 1.0          # interpolation resolution for spectrum in km/s
speed_of_light = scc.c / 1000       # in km/s

spectrum_filenames = ['FIBj030100_step011_merge.fits', 'FIBj030108_step011_merge.fits', 'FIBj040099_step011_merge.fits']
radial_velocity_A = [15.66729, 15.86979, 18.95343]
data_path = '../Data/unprocessed/NOT/KIC8430105/'
spectra_wavelength_collection = []
spectra_flux_collection = []
spectra_barycorr = []

for filename in spectrum_filenames:
    wavelength, flux, date, ra, dec = spf.load_program_spectrum(data_path + filename)
    ra = ra*15.0

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

    # Append to collection
    spectra_wavelength_collection.append(wavelength)
    spectra_flux_collection.append(flux)

    # Calculate barycentric correction
    time = Time(date, scale='utc', location=observatory_location)
    time.format = 'jd'
    time.out_subfmt = 'long'
    bc_rv_cor, _, _ = get_BC_vel(time, ra=ra, dec=dec, starname=stellar_target, ephemeris='de432s',
                                       obsname=observatory)
    spectra_barycorr.append(bc_rv_cor/1000)

# # Resample all spectra to the same wavelength grid, equi-spaced in velocity space # #
wavelength, flux_collection_array = spf.resample_multiple_spectra(
    delta_v, (spectra_wavelength_collection, spectra_flux_collection)
)

flux_collection_array = flux_collection_array[0]
print(type(flux_collection_array))
print(len(flux_collection_array))
# # Perform barycentric and RV correction # #
for i in range(0, flux_collection_array[0, :].size):
    flux_collection_array[:, i] = ssr.shift_spectrum(
        flux_collection_array[:, i], spectra_barycorr[i] - radial_velocity_A[i], delta_v
    )

# # Plot # #
plt.figure()
for i in range(0, len(flux_collection_array[0, :])):
    plt.plot(wavelength, flux_collection_array[:, i])

# # Calculate co-added spectrum (mean) # #
flux_mean = np.zeros(shape=(flux_collection_array[:, 0].size, ))
for i in range(0, len(flux_collection_array[0, :])):
    flux_mean += flux_collection_array[:, i]
flux_mean = flux_mean / flux_collection_array[0, :].size

# # Plot again # #
plt.figure()
plt.plot(wavelength, flux_mean)
plt.show()

# # Save co-added spectrum # #
save_array = np.empty((wavelength.size, 5))
save_array[:, 0] = wavelength
save_array[:, 1] = flux_mean
save_array[:, 2] = flux_collection_array[:, 0]
save_array[:, 3] = flux_collection_array[:, 1]
save_array[:, 4] = flux_collection_array[:, 2]
np.savetxt(
    'coadded_eclipse_spectrum.txt', save_array, delimiter='\t',
    header='wavelength [Å]\tcoadded spectrum\tspectrum 1\tspectrum 2\tspectrum 3'
)
