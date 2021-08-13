from matplotlib import pyplot as plt; import numpy as np; import os
import RV.library.spectrum_processing_functions as spf
from astropy.time import Time
from astropy.coordinates import EarthLocation
from barycorrpy import utc_tdb

observatory_location = EarthLocation.of_site("lapalma")
observatory_name = "lapalma"
stellar_target = "kic8430105"


spectra_folder_path = 'Data/unprocessed/NOT/KIC8430105/'
reduced_spectra_folder_path = 'Data/processed/AFS_algorithm/Normalized_Spectrum/'
filename_identifier = 'merge.fits'
reduced_filename_identifier = '_reduced_set.dat'

filename_list = []
flux_collection_list = []
wavelength_collection_list = []
date_array = []
RA_array = np.array([])
DEC_array = np.array([])

period = 63.33

eclipse_primary = 0.0
eclipse_secondary = 0.341
approximate_eclipse_hwidth= 0.02


for filename in os.listdir(spectra_folder_path):
    if filename_identifier in filename and '.lowSN' not in filename:
        _, _, date, ra, dec = spf.load_program_spectrum(spectra_folder_path + filename)

        filename_bulk = filename[:filename.rfind(".fits")]
        wavelength, flux = np.loadtxt(reduced_spectra_folder_path+filename_bulk+reduced_filename_identifier,
                                      unpack=True)

        # Remove values over 1.05 and under 0
        selection_mask = (flux > 1.1) | (flux < 0.0)
        flux = flux[~selection_mask]
        wavelength = wavelength[~selection_mask]

        wavelength_collection_list.append(wavelength)
        flux_collection_list.append(flux)
        date_array = np.append(date_array, date)
        RA_array = np.append(RA_array, ra * 15.0)  # converts unit
        DEC_array = np.append(DEC_array, dec)
        filename_list.append(filename[:filename.rfind("_step011_merge.fits")])


# # Calculate bjdtdb
RA, DEC = RA_array[0], DEC_array[0]
times = Time(date_array, scale='utc', location=observatory_location)
times.format = 'jd'
times.out_subfmt = 'long'
bjdtdb, _, _ = utc_tdb.JDUTC_to_BJDTDB(times, ra=RA, dec=DEC, starname=stellar_target, obsname=observatory_name)

# # Extra observation date
time_extra = Time('2021-08-02T23:03:19', scale='utc', location=observatory_location)
time_extra.format = 'jd'
time_extra.out_subfmt = 'long'
bjd_extra, _, _ = utc_tdb.JDUTC_to_BJDTDB(time_extra, ra=RA, dec=DEC, starname=stellar_target, obsname=observatory_name)


# Phases
model_filename = '../Binary_Analysis/JKTEBOP/kepler_LTF/model.out'
bjdtdb -= 2400000 + 54976.6348
phase_model, rv_Bm, rv_Am = np.loadtxt(model_filename, usecols=(0, 6, 7), unpack=True)
phase_spectra = np.mod(bjdtdb, period) / period

# Sort
sort_idx = np.argsort(phase_spectra)
filenames_sorted = np.array(filename_list)[sort_idx]
phase_spectra_sorted = phase_spectra[sort_idx]

# RV plot
fig_rv = plt.figure(figsize=(16, 9))
gs_rv = fig_rv.add_gridspec(1, 1)
ax_rv = fig_rv.add_subplot(gs_rv[:, :])


ax_rv.plot(phase_model, rv_Am-16.053, 'b')
ax_rv.plot(phase_model, rv_Bm-16.053, 'r')
ax_rv.plot(phase_spectra, np.zeros(phase_spectra.shape), 'k*', markersize=5)
for i in range(0, filenames_sorted.size):
    ax_rv.annotate(filenames_sorted[i], (phase_spectra_sorted[i], 0),
                   (phase_spectra_sorted[i], (np.mod(i, filenames_sorted.size//4)/(filenames_sorted.size//4) - 0.5)*30),
                   arrowprops={'arrowstyle': '->'})
ax_rv.set_xlabel('Orbital Phase')

ax_rv.plot([eclipse_primary, eclipse_primary], [-40, 50], color='gray')
ax_rv.plot([1.0-approximate_eclipse_hwidth, 1.0-approximate_eclipse_hwidth], [-40, 50], '--',
           color='gray')

ax_rv.plot([eclipse_primary+approximate_eclipse_hwidth, eclipse_primary+approximate_eclipse_hwidth], [-40, 50], '--',
           color='gray')
ax_rv.plot([eclipse_secondary, eclipse_secondary], [-40, 50], color='gray')
ax_rv.plot([eclipse_secondary-approximate_eclipse_hwidth, eclipse_secondary-approximate_eclipse_hwidth], [-40, 50],
           '--', color='gray')
ax_rv.plot([eclipse_secondary+approximate_eclipse_hwidth, eclipse_secondary+approximate_eclipse_hwidth], [-40, 50],
           '--', color='gray')
ax_rv.plot(np.mod(bjd_extra - 2454976.6348, period)/period, 0, 'g*')

plt.show(block=True)

# Spectrum plots
for i in range(0, filenames_sorted.size, 2):
    fig_spectra = plt.figure(figsize=(16, 9))
    gs_spectra = fig_spectra.add_gridspec(1, 2)
    ax1 = fig_spectra.add_subplot(gs_spectra[:, 0])
    ax2 = fig_spectra.add_subplot(gs_spectra[:, 1])

    ax1.plot(wavelength_collection_list[sort_idx[i]], flux_collection_list[sort_idx[i]])
    ax2.plot(wavelength_collection_list[sort_idx[i+1]], flux_collection_list[sort_idx[i+1]])

    ax1.set_title(filenames_sorted[i])
    ax2.set_title(filenames_sorted[i+1])
    ax1.set_xlabel('Wavelength [Å]')
    ax2.set_xlabel('Wavelength [Å]')

    ax1.set_xlim([4700, 5400])
    ax2.set_xlim([4700, 5400])
    plt.show(block=False)

plt.show(block=True)