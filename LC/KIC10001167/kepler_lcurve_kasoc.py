import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.io import fits
from numpy.polynomial import Polynomial


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
# Correct flux while keeping transits
flux_transit = flux / (corr_long + corr_short)
# Do phase fold
period = 120.3903
phase = np.mod(time, period) / period

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
flux_transit = flux_transit[~mask]
phase = phase[~mask]


if False:
    plt.figure()
    plt.plot(time, corr_long, 'g.', markersize=0.5)
    plt.plot(time, flux, 'b.', markersize=0.2)
    plt.legend(['F_long', 'Flux'])
    plt.xlabel('BJD - 2400000')
    plt.ylabel('e-/s')
    plt.show(block=False)

    plt.figure()
    plt.plot(time, corr_transit, 'r.', markersize=1)
    plt.plot(time, corr_short, 'b.', markersize=0.5)
    plt.legend(['F_phase', 'F_short'])
    plt.xlabel('BJD - 2400000')
    plt.ylabel('e-/s')
    plt.show(block=False)


if False:
    _, axs = plt.subplots(2, 2, sharey='row', figsize=(17.78, 10))
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]
    ax1.plot(phase, flux / corr_long, 'r.', markersize=0.7)
    ax1.set_xlim([0.04, 0.24])
    ax1.set_ylabel('Relative Flux')
    ax2.plot(phase, flux / (corr_long + corr_short), 'b.', markersize=0.7)
    ax2.set_xlim([0.04, 0.24])
    ax3.plot(phase, flux / corr_long, 'r.', markersize=0.7)
    ax3.set_xlim([0.38, 0.58])
    ax3.legend([r'$\frac{x}{x_{long}}$'], markerscale=15)
    ax3.set_ylabel('Relative Flux')
    ax3.set_xlabel('Phase')
    ax4.plot(phase, flux / (corr_long + corr_short), 'b.', markersize=0.7)
    ax4.legend([r'$\frac{x}{x_{long}+x_{short}}$'], markerscale=15)
    ax4.set_xlim([0.38, 0.58])
    ax4.set_xlabel('Phase')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.savefig(fname='../../figures/report/kasoc/10001167/shortlong.png', dpi=400)
    plt.savefig(fname='../../figures/report/kasoc/10001167/shortlong.pdf', dpi=300)
    plt.show(block=False)

if False:
    plt.figure()
    plt.plot(phase, corr_short, 'r.', markersize=1)
    plt.plot(phase, corr_transit, 'b.', markersize=0.5)
    plt.legend(['F_short', 'F_phase'])
    plt.ylim([np.min(corr_short[~np.isnan(corr_short)]), np.max(corr_short[~np.isnan(corr_short)])])
    plt.xlabel('Phase')
    plt.ylabel('e-/s')
    plt.show(block=False)
    corr_long_plot = corr_long / np.median(corr_long)
    plt.figure()
    plt.plot(phase, corr_long_plot, 'r.', markersize=1)
    plt.plot(phase, 1 - corr_transit / np.min(corr_transit), 'b.', markersize=0.5)
    plt.legend(['F_long', 'F_phase'])
    plt.ylim([np.min(corr_long_plot[~np.isnan(corr_long_plot)]), np.max(corr_long_plot[~np.isnan(corr_long_plot)])])
    plt.xlabel('Phase')
    plt.ylabel('Normalized filter')
    plt.show(block=True)

if False:
    plt.figure(figsize=(17.78, 10))
    plt.plot(phase, corr_transit, '.', markersize=7)
    plt.xlabel('Phase')
    plt.ylabel('e-/s')
    plt.xlim([0, 0.6])
    plt.tight_layout()
    plt.savefig(fname='../../figures/report/kasoc/10001167/xphase.png', dpi=400)
    plt.savefig(fname='../../figures/report/kasoc/10001167/xphase.pdf', dpi=300)
    plt.show()


# # # Exclude bad data regions found by LTF in other data reduction method # # #
# mask_bad_data = ((time > 54990) & (time < 55008)) | ((time > 55117) & (time < 55136)) | \
#                 ((time > 55223) & (time < 55240)) | ((time > 55560) & (time < 55575)) | \
#                 ((time > 56383) & (time < 56403))

# if False:
#     plt.figure()
#     plt.plot(time, flux_transit, 'r.', markersize=1.5)
#     plt.plot(time[~mask_bad_data], flux_transit[~mask_bad_data], 'b.', markersize=1)
#     plt.xlim([56373, 56413])
#     plt.show()

# time, flux, flux_transit, phase = time[~mask_bad_data], flux[~mask_bad_data], flux_transit[~mask_bad_data], \
#                                   phase[~mask_bad_data]

# # # Convert to magnitudes # # #
m = -2.5*np.log10(flux_transit)
# Measure spread within full eclipse as estimator of flux error
plt.plot(phase, flux_transit)
plt.show()
rmse_measure_mask = (phase > 0.483) & (phase < 0.5088)
rmse_used_vals = flux_transit[rmse_measure_mask]
mean_val = np.mean(rmse_used_vals)
error = np.sqrt(np.sum((mean_val - rmse_used_vals)**2) / rmse_used_vals.size)
m_err = np.abs(-2.5/np.log(10) * (error/flux_transit))
# m_err = np.abs(-2.5/np.log(10) * ((err_seism * 1E-6)*(corr_full/(corr_long+corr_short)))/flux_transit)
if False:
    plt.figure()
    plt.errorbar(time, m, m_err, fmt='k.', markersize=0.5, elinewidth=0.5)
    plt.ylim([0.020, -0.003])
    plt.show()

# # # Fold lightcurve and cut off data away from eclipses # # #
# x0 = 0.117
# x1 = 0.156
# x2 = 0.457
# x3 = 0.500

# print(x0, x1, x2, x3)
# mask = ((phase>=x0) & (phase<=x1)) | ((phase>=x2) & (phase<=x3))

# m_ = m[mask]
# time_ = time[mask]
# m_err_ = m_err[mask]
# phase_ = phase[mask]

if True:
    _, ax1 = plt.subplots(1, 1, figsize=(16, 9))
    print('phase.size', phase.size)
    ax1.errorbar(phase, m, m_err, fmt='k.', ecolor='darkgray', markersize=0.9, elinewidth=0.4, errorevery=3)
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Relative Magnitude')
    ax1.set_ylim([0.027, -0.007])
    plt.tight_layout()
    plt.savefig(fname='../../figures/report/kasoc/10001167/mag_uncorr.png', dpi=400)
    plt.savefig(fname='../../figures/report/kasoc/10001167/mag_uncorr.pdf', dpi=300)
    plt.show()

if False:
    _, ax1 = plt.subplots(1, 1)
    ax1.errorbar(phase, m, m_err, fmt='k.', ecolor='gray', markersize=0.5, elinewidth=0.1, errorevery=10)
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Relative Magnitude')
    plt.show()

# # # Save fit lightcurve to file # # #
# mask = ~np.isnan(m_) & ~np.isnan(m_err_) & ~np.isnan(time_)
# m_ = m_[mask]
# time_ = time_[mask]
# m_err_ = m_err_[mask]

# save_data = np.zeros((m_.size, 3))
# save_data[:, 0] = time_
# save_data[:, 1] = m_
# save_data[:, 2] = m_err_
# np.savetxt('Data/processed/KIC10001167/lcmag_kepler.txt', save_data, header='Time\tMagnitude\tError', delimiter='\t')

# # # Save full lightcurve to file # # #
mask = ~np.isnan(m) & ~np.isnan(m_err) & ~np.isnan(time)
m = m[mask]
time = time[mask]
m_err = m_err[mask]

save_data = np.zeros((m.size, 3))
save_data[:, 0] = time
save_data[:, 1] = m
save_data[:, 2] = m_err
np.savetxt('../Data/processed/KIC10001167/lcmag_kasoc_full.txt', save_data, header='Time\tMagnitude\tError', delimiter='\t')
save_data[:, 1] = flux_transit
save_data[:, 2] = error
np.savetxt('../Data/processed/KIC10001167/lcflux_kasoc_full.txt', save_data, header='Time\tMagnitude\tError', delimiter='\t')

