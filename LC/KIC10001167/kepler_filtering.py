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

plt.figure()
plt.plot(time, flux)

# Create plots of individual phases
n_orbits = int(np.ceil((np.max(time) - np.min(time))/period))
fig, axs = plt.subplots(nrows=n_orbits, figsize=(8, 9), squeeze=True)
for i in range(0, n_orbits):
    ax = axs[i]
    ax.set_yticks([])
    ax.set_xticks([])
    orbit_start = np.min(time) + i*period
    orbit_mask = (time >= orbit_start) & (time < orbit_start + period)
    orbit_phase = phase[orbit_mask]
    orbit_flux = flux[orbit_mask]
    kasoc_long = corr_long[orbit_mask]

    phase_split_idx = np.argwhere(np.diff(orbit_phase) < -0.5)
    if phase_split_idx:
        print(phase_split_idx)
        phase_1 = orbit_phase[0:phase_split_idx[0][0]+1]
        flux_1 = orbit_flux[0:phase_split_idx[0][0]+1]
        klong_1 = kasoc_long[0:phase_split_idx[0][0]+1]
        phase_2 = orbit_phase[phase_split_idx[0][0] + 1:]
        flux_2 = orbit_flux[phase_split_idx[0][0] + 1:]
        klong_2 = kasoc_long[phase_split_idx[0][0] + 1:]
        ax.plot(phase_1, flux_1, 'r')
        ax.plot(phase_2, flux_2, 'r')
        ax.plot(phase_1, klong_1, 'k--', linewidth=1)
        ax.plot(phase_2, klong_2, 'k--', linewidth=1)
    else:
        ax.plot(orbit_phase, orbit_flux, 'r')
        ax.plot(orbit_phase, kasoc_long, 'k--', linewidth=1)
    ax.set_xlim([0.0, 1.0])
print(n_orbits)
axs[-1].set_xlabel('Orbital Phase')
axs[-1].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

# Long median filter
cadence = 0.02083   # 30 minutes in days
period_size = int(np.round(period/cadence))
filter_long = median_filter(flux, size=3*period_size, mode='reflect')
plt.plot(time, filter_long)
plt.plot(time, corr_long)
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

print(np.mean(np.diff(time)))

# Long median filter
cadence = 0.02083   # 30 minutes in days
period_size = int(np.round(period/cadence))
plt.figure()
plt.title('filter_3p and flux')
filter_long = median_filter(flux, size=3*period_size, mode='reflect')
plt.plot(time, filter_long)
plt.plot(time, flux)
plt.show(block=False)

# Create plots of individual phases
n_orbits = int(np.ceil((np.max(time) - np.min(time))/period))
fig, axs = plt.subplots(nrows=n_orbits, figsize=(8, 9), squeeze=True)
for i in range(0, n_orbits):
    ax = axs[i]
    ax.set_yticks([])
    ax.set_xticks([])
    orbit_start = np.min(time) + i*period
    orbit_mask = (time >= orbit_start) & (time < orbit_start + period)
    orbit_phase = phase[orbit_mask]
    orbit_flux = flux[orbit_mask]
    kasoc_long = filter_long[orbit_mask]

    phase_split_idx = np.argwhere(np.diff(orbit_phase) < -0.5)
    if phase_split_idx:
        print(phase_split_idx)
        phase_1 = orbit_phase[0:phase_split_idx[0][0]+1]
        flux_1 = orbit_flux[0:phase_split_idx[0][0]+1]
        klong_1 = kasoc_long[0:phase_split_idx[0][0]+1]
        phase_2 = orbit_phase[phase_split_idx[0][0] + 1:]
        flux_2 = orbit_flux[phase_split_idx[0][0] + 1:]
        klong_2 = kasoc_long[phase_split_idx[0][0] + 1:]
        ax.plot(phase_1, flux_1, 'r')
        ax.plot(phase_2, flux_2, 'r')
        ax.plot(phase_1, klong_1, 'k--', linewidth=1)
        ax.plot(phase_2, klong_2, 'k--', linewidth=1)
    else:
        ax.plot(orbit_phase, orbit_flux, 'r')
        ax.plot(orbit_phase, kasoc_long, 'k--', linewidth=1)
    ax.set_xlim([0.0, 1.0])
print(n_orbits)
axs[-1].set_xlabel('Orbital Phase')
axs[-1].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

# Short filter KASOC
plt.figure()
plt.plot(phase, corr_short, '.', markersize=1)
plt.title('KASOC short filter')


n_orbits = int(np.ceil((np.max(time) - np.min(time))/period))
fig, axs = plt.subplots(nrows=n_orbits, figsize=(8, 9), squeeze=True)
axs[0].set_title('KASOC long filter / filter_long')
for i in range(0, n_orbits):
    ax = axs[i]
    ax.set_yticks([])
    ax.set_xticks([])
    orbit_start = np.min(time) + i*period
    orbit_mask = (time >= orbit_start) & (time < orbit_start + period)
    orbit_phase = np.mod(time, period)/period  # phase[orbit_mask]
    orbit_phase = orbit_phase[orbit_mask]
    kasoc_long = corr_long[orbit_mask]
    flong = filter_long[orbit_mask]
    phase_split_idx = np.argwhere(np.diff(orbit_phase) < -0.5)
    if phase_split_idx:
        print(phase_split_idx)
        phase_1 = orbit_phase[0:phase_split_idx[0][0]+1]
        div_1 = kasoc_long[0:phase_split_idx[0][0]+1]/flong[0:phase_split_idx[0][0]+1]
        phase_2 = orbit_phase[phase_split_idx[0][0] + 1:]
        div_2 = kasoc_long[phase_split_idx[0][0] + 1:]/flong[phase_split_idx[0][0] + 1:]
        ax.plot(phase_1, div_1, 'r')
        ax.plot(phase_2, div_2, 'r')
    else:
        ax.plot(orbit_phase, kasoc_long/flong, 'r')
    ax.set_xlim([0.0, 1.0])
print(n_orbits)
axs[-1].set_xlabel('Orbital Phase')
axs[-1].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# Long trend corrected
plt.figure()
plt.title('flux/filter_3p')
plt.plot(time, flux/filter_long)
plt.figure()
plt.title('flux/filter_3p and flux/(kasoc_long+kasoc_short)')
plt.plot(phase, flux/filter_long, '.', markersize=1)
plt.plot(phase, flux_kasoc, '.', markersize=1)
plt.show(block=False)

# Create transit filter
transit_filter = median_filter(flux/filter_long, size=int(2.0*period_size), mode='reflect')
plt.figure()
plt.title('flux/filter_3p and filter_2p')
plt.plot(time, flux/filter_long)
plt.plot(time, transit_filter)
plt.show(block=False)
plt.figure()
plt.title('filter_2p')
plt.plot(phase, transit_filter)

plt.figure()
plt.title('flux/filter_3p and flux/filter_3p /filter_2p')
plt.plot(phase, (flux/filter_long), '.', markersize=1)
plt.plot(phase, (flux/filter_long)/transit_filter, 'r.', markersize=1)
plt.figure()
plt.title('(flux/filter_3p)/filter_2p and flux/(kasoc_long+kasoc_short)')
plt.plot(phase, (flux/filter_long)/transit_filter, '.', markersize=1)
plt.plot(phase, flux_kasoc, '.', markersize=1)
plt.figure()
plt.title('(flux/filter_3p)/filter_2p and flux/(kasoc_long+kasoc_short)')
plt.plot(time, (flux/filter_long)/transit_filter, '.', markersize=1)
plt.plot(time, flux_kasoc, '.', markersize=1)
plt.show()