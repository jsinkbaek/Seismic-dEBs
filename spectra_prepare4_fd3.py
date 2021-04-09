import numpy as np
from astropy.io import fits
import os
from scipy.interpolate import interp1d

"""
This script is used to prepare observed (reduced) spectra from NOT in the form of .fits files into a format that can be
used by fd3. This process includes interpolating all the spectra to the same wavelength grid.
"""

# # Set variables for script # #
data_path = 'datafiles/NOT/KIC8430105/'
masterfile_name = 'not_kic8430105.master.obs'

# # Prepare lists and arrays # #
yvals_list = []
wl_list = []
wl_end = np.array([])
wl_start = np.array([])
delta_wl_array = np.array([])

# # Load fits files and save data # #
for filename in os.listdir(data_path):
    if '.fits' in filename and '.lowSN' not in filename:
        with fits.open(data_path + filename) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data
            wl0 = hdr['CRVAL1']
            delta_wl = hdr['CDELT1']
            date = hdr['DATE_OBS']

        wl = np.linspace(wl0, wl0+delta_wl*data.size, data.size)
        wl_list.append(wl)
        yvals_list.append(data)
        wl_end = np.append(wl_end, wl[-1])
        wl_start = np.append(wl_start, wl0)
        delta_wl_array = np.append(delta_wl_array, delta_wl)

# # Set unified wavelength grid # #
wl0_unified = np.max(wl_start)
wl1_unified = np.min(wl_end)
delta_wl_unified = np.average(delta_wl_array)*100
wl_unified = np.arange(wl0_unified, wl1_unified, delta_wl_unified)

# # Interpolate data to wavelength grid # #
yvals_new = []
for i in range(0, len(wl_list)):
    f_intp = interp1d(wl_list[i], yvals_list[i], kind='cubic')
    yvals_new.append(f_intp(wl_unified))

# # Convert to wavelength logarithm # #
log_wl = np.log(wl_unified)

# # Save spectrum data # #
save_data = np.empty(shape=(log_wl.size, len(yvals_new)+1))
save_data[:, 0] = log_wl
for i in range(0, len(yvals_new)):
    save_data[:, i+1] = yvals_new[i]
np.savetxt(data_path+masterfile_name, save_data)


# # # Calculate Barycentric Correction and BJD # # #

