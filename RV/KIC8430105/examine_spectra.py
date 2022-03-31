from matplotlib import pyplot as plt; import numpy as np; import os
import RV.library.spectrum_processing_functions as spf
from astropy.time import Time
from astropy.coordinates import EarthLocation
from barycorrpy import utc_tdb

target = "kic8430105"
observatory_name = 'Roque de los Muchachos'
observatory_location = EarthLocation.of_site(observatory_name)
stellar_target = "kic8430105"


spectra_folder_path = '../Data/unprocessed/NOT/KIC8430105/'
filename_identifier = 'merge.fits'
reduced_filename_identifier = '_reduced_set.dat'

filename_list = []
date_array = []
RA_array = np.array([])
DEC_array = np.array([])
sn_array = np.array([])

period = 63.3271045716

eclipse_primary = 0.0
eclipse_secondary = 0.6589
approximate_eclipse_hwidth= 0.02

wavelength_normalization_limit = (4450, 7000)

for filename in os.listdir(spectra_folder_path):
    if filename_identifier in filename and '.lowSN' not in filename:
        wavelength, flux, date, ra, dec = spf.load_program_spectrum(spectra_folder_path + filename)

        filename_bulk = filename[:filename.rfind(".fits")]

        date_array = np.append(date_array, date)
        RA_array = np.append(RA_array, ra * 15.0)  # converts unit
        DEC_array = np.append(DEC_array, dec)
        filename_list.append(filename[:filename.rfind("_step011_merge.fits")])

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

        # Calculate S/N
        mask = (wavelength > 5605) & (wavelength < 5613)
        rms = np.sqrt(np.sum((1-flux[mask]) ** 2)/flux[mask].size)
        signal_noise = 1/rms
        # print(rms)
        # print(signal_noise)

        plt.figure(figsize=(16, 9))
        plt.plot(wavelength, flux)
        plt.plot(wavelength[mask], flux[mask])
        plt.xlim([5602, 5617])
        sn_array = np.append(sn_array, signal_noise)

plt.show()


# # Calculate bjdtdb
RA, DEC = np.mean(RA_array), np.mean(DEC_array)
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
model_filename = '../../Binary_Analysis/JKTEBOP/NOT/kepler_pdcsap/model.out'
rvA_filename = '../../Binary_Analysis/JKTEBOP/NOT/kepler_pdcsap/rvA.out'
rvB_filename = '../../Binary_Analysis/JKTEBOP/NOT/kepler_pdcsap/rvB.out'
bjdtdb -= 2400000 + 54998.2336737188  # 54976.6348
phase_model, rv_Am, rv_Bm = np.loadtxt(model_filename, usecols=(0, 6, 7), unpack=True)
phase_spectra = np.mod(bjdtdb, period) / period
time_rvA, rvA, error_rvA, phase_rvA = np.loadtxt(rvA_filename, usecols=(0, 1, 2, 3), unpack=True)
time_rvB, rvB, error_rvB, phase_rvB = np.loadtxt(rvB_filename, usecols=(0, 1, 2, 3), unpack=True)
sort_idx = np.argsort(phase_rvA)
time_rvA, rvA, error_rvA, phase_rvA = time_rvA[sort_idx], rvA[sort_idx], error_rvA[sort_idx], phase_rvA[sort_idx]
sort_idx = np.argsort(phase_rvB)
time_rvB, rvB, error_rvB, phase_rvB = time_rvB[sort_idx], rvB[sort_idx], error_rvB[sort_idx], phase_rvB[sort_idx]

# Sort
sort_idx = np.argsort(phase_spectra)
filenames_sorted = np.array(filename_list)[sort_idx]
phase_spectra_sorted = phase_spectra[sort_idx]
sn_sorted = sn_array[sort_idx]

print('Phase\tSpectra\tS/N')
for i in range(0, len(phase_spectra_sorted)):
    print(phase_spectra_sorted[i], filenames_sorted[i], sn_sorted[i])
print('')
print('Processed rv A:')
print('Phase\tTime\tRV\tError')
for i in range(0, len(rvA)):
    print(phase_rvA[i], time_rvA[i]-50000, rvA[i], error_rvA[i])
print('')
print('Processed rv B:')
print('Phase\tTime\tRV\tError')
for i in range(0, len(rvB)):
    print(phase_rvB[i], time_rvB[i]-50000, rvB[i], error_rvB[i])
print('')
print('Phase\tSpectra')
for i in range(0, len(phase_spectra_sorted)):
    print(phase_spectra_sorted[i], filenames_sorted[i])
# RV plot
fig_rv = plt.figure(figsize=(16, 9))
gs_rv = fig_rv.add_gridspec(1, 1)
ax_rv = fig_rv.add_subplot(gs_rv[:, :])

system_rvA = 11.6144
system_rvB = 12.0293

ax_rv.plot(phase_model, rv_Am-system_rvA, 'r')
ax_rv.plot(phase_model, rv_Bm-system_rvB, 'b')
ax_rv.plot(phase_rvA, rvA-system_rvA, 'r*')
ax_rv.plot(phase_rvB, rvB-system_rvB, 'b*')
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
ax_rv.plot(np.mod(bjd_extra - 2454998.2336737188, period)/period, 0, 'g*')

plt.show(block=True)
