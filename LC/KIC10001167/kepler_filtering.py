import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.io import fits
from numpy.polynomial import Polynomial
from scipy.ndimage import median_filter



# # # Start of Script # # #
matplotlib.use('Qt5Agg')
matplotlib.rcParams.update({'font.size': 25})
# Load data
with fits.open("../Data/unprocessed/kasoc/kplr010001167_kasoc-ts_llc_v2.fits") as hdul:
    print(hdul.info())
    hdu = hdul[1]
    print(hdu.columns)
    time = hdu.data['TIME']
    flux_seism = hdu.data['FLUX']
    err_seism = hdu.data['FLUX_ERR']
    corr_long = hdu.data['XLONG']
    corr_transit = hdu.data['XPHASE']
    corr_short = hdu.data['XSHORT']
    corr_pos = hdu.data['XPOS']
    corr_full = hdu.data['FILTER']
    kasoc_qflag = hdu.data['KASOC_FLAG']

# Remove nan's
nan_mask = np.isnan(time) | np.isnan(flux_seism) | np.isnan(corr_full)
time = time[~nan_mask]
flux_seism = flux_seism[~nan_mask]
err_seism = err_seism[~nan_mask]
corr_long = corr_long[~nan_mask]
corr_transit = corr_transit[~nan_mask]
corr_short = corr_short[~nan_mask]
corr_full = corr_full[~nan_mask]
corr_pos = corr_pos[~nan_mask]

# Make uncorrected relative flux
flux = (flux_seism * 1E-6 + 1) * corr_full
# Do phase fold
period = 120.3903
phase = np.mod(time, period) / period

# Long median filter
cadence = 0.02083   # 30 minutes in days
period_size = int(np.round(period/cadence))
filter_long = median_filter(flux, size=2*period_size, mode='reflect')
plt.plot(time, filter_long)
plt.plot(time, flux)

# Clip out thermal relaxation part
mask = (time > 55068) & (time < 55110)
time = time[~mask]
flux_seism = flux_seism[~mask]
err_seism = err_seism[~mask]
corr_long = corr_long[~mask]
corr_transit = corr_transit[~mask]
corr_short = corr_short[~mask]
corr_full = corr_full[~mask]
corr_pos = corr_pos[~mask]
flux = flux[~mask]
phase = phase[~mask]

flux_kasoc = flux / (corr_long + corr_short)

# Move pre-relaxed dataset down
mask_pre = (time < 55068) & (time > 55064)
mask_post = (time > 55100) & (time < 55126)
avg_pre = np.mean(flux[mask_pre])
avg_post = np.mean(flux[mask_post])
offset = avg_post - avg_pre
mask_pre_all = (time < 55068)
flux[mask_pre_all] += offset

# Long median filter
cadence = 0.02083   # 30 minutes in days
period_size = int(np.round(period/cadence))
filter_long = median_filter(flux, size=2*period_size, mode='reflect')
plt.plot(time, filter_long)
plt.plot(time, flux)
plt.show(block=False)

# Long trend corrected
plt.figure()
plt.plot(time, flux/filter_long)
plt.figure()
plt.plot(phase, flux/filter_long, '.', markersize=1)
plt.plot(phase, flux_kasoc, '.', markersize=1)
plt.show(block=False)

# Create transit filter
transit_filter = median_filter(flux/filter_long, size=period_size, mode='reflect')
plt.figure()
plt.plot(time, flux/filter_long)
plt.plot(time, transit_filter)
plt.show(block=False)

plt.figure()
plt.plot(time, (flux/filter_long)/transit_filter)
plt.figure()
plt.plot(phase, (flux/filter_long)/transit_filter, '.', markersize=1)
plt.plot(phase, flux_kasoc, '.', markersize=1)
plt.show()