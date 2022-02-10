import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.io import fits
from numpy.polynomial import Polynomial


# # # Start of Script # # #
# matplotlib.use('Qt5Agg')
matplotlib.rcParams.update({'font.size': 25})
# Load data
with fits.open("../Data/unprocessed/kasoc/kplr008430105_kasoc-ts_llc_v1.fits") as hdul:
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
period = 63.32713
phase = np.mod(time, period) / period


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
    plt.show(block=True)


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
    plt.savefig(fname='../figures/report/kasoc/shortlong.png', dpi=400)
    plt.savefig(fname='../figures/report/kasoc/shortlong.pdf', dpi=300)
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

if True:
    plt.figure(figsize=(17.78, 10))
    plt.plot(phase, corr_transit, '.', markersize=7)
    plt.xlabel('Phase')
    plt.ylabel('e-/s')
    plt.xlim([0, 0.6])
    plt.tight_layout()
    plt.savefig(fname='../../figures/report/kasoc/xphase.png', dpi=400)
    plt.savefig(fname='../../figures/report/kasoc/xphase.pdf', dpi=300)
    plt.show()


# # # Exclude bad data regions found by LTF in other data reduction method # # #
mask_bad_data = ((time > 54990) & (time < 55008)) | ((time > 55117) & (time < 55136)) | \
                ((time > 55223) & (time < 55240)) | ((time > 55560) & (time < 55575)) | \
                ((time > 56383) & (time < 56403))

if False:
    plt.figure()
    plt.plot(time, flux_transit, 'r.', markersize=1.5)
    plt.plot(time[~mask_bad_data], flux_transit[~mask_bad_data], 'b.', markersize=1)
    plt.xlim([56373, 56413])
    plt.show()

time, flux, flux_transit, phase = time[~mask_bad_data], flux[~mask_bad_data], flux_transit[~mask_bad_data], \
                                  phase[~mask_bad_data]

# # # Convert to magnitudes # # #
m = -2.5*np.log10(flux_transit)
# Measure spread within full eclipse as estimator of flux error
rmse_measure_mask = (phase > 0.126) & (phase < 0.149)
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
x0 = 0.117
x1 = 0.156
x2 = 0.457
x3 = 0.500
# x0=0.076
# x1=0.20
# x2=0.42
# x3=0.53
print(x0, x1, x2, x3)
mask = ((phase>=x0) & (phase<=x1)) | ((phase>=x2) & (phase<=x3))

m_ = m[mask]
time_ = time[mask]
m_err_ = m_err[mask]
phase_ = phase[mask]

if True:
    plt.figure()
    plt.errorbar(phase, m, m_err, fmt='.', color='gray', markersize=0.5, elinewidth=0.4)
    plt.errorbar(phase_, m_, m_err_, fmt='k.', markersize=0.5, elinewidth=0.5)
    plt.xlabel('Phase')
    plt.ylabel('Relative Magnitude')
    plt.legend(['Excluded data', 'Included data'])
    plt.ylim([0.020, -0.003])
    plt.show()

if True:
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(17.78, 10))
    print('phase.size', phase.size)
    ax1.errorbar(phase, m, m_err, fmt='k.', ecolor='darkgray', markersize=0.9, elinewidth=0.4, errorevery=3)
    ax1.set_xlim([0.04, 0.24])
    ax1.set_xlabel('Orbital Phase', fontsize=22)
    ax1.set_ylabel('Relative Magnitude', fontsize=22)
    ax1.annotate('', [0.1153, 0.00105], xytext=[0.1017, 0.00321], arrowprops={'arrowstyle': '->'})
    ax1.annotate('', [0.1615, 0.00105], xytext=[0.1670, 0.00382], arrowprops={'arrowstyle': '->'})
    ax2.annotate('', [0.4543, 0.0010], xytext=[0.4476, 0.00292], arrowprops={'arrowstyle': '->'})
    ax2.annotate('', [0.4975, 0.00092], xytext=[0.5067, 0.00285], arrowprops={'arrowstyle': '->'})
    ax1.set_ylim([0.020, -0.003])
    ax2.errorbar(phase, m, m_err, fmt='k.', ecolor='darkgray', markersize=0.9, elinewidth=0.4, errorevery=3)
    ax2.set_xlim([0.38, 0.58])
    ax2.set_xlabel('Orbital Phase', fontsize=22)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.savefig(fname='../../figures/report/kasoc/mag_uncorr.png', dpi=400)
    plt.savefig(fname='../../figures/report/kasoc/mag_uncorr.pdf', dpi=300)
    plt.show()

if True:
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.errorbar(phase_, m_, m_err_, fmt='k.', ecolor='gray', markersize=0.5, elinewidth=0.1, errorevery=10)
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Relative Magnitude')
    ax1.set_xlim([0.115, 0.158])
    ax1.set_ylim([0.020, -0.003])
    ax2.errorbar(phase_, m_, m_err_, fmt='k.', ecolor='gray', markersize=0.5, elinewidth=0.1, errorevery=10)
    ax2.set_xlabel('Phase')
    ax2.set_xlim([0.455, 0.502])
    plt.show()

# # # Save fit lightcurve to file # # #
mask = ~np.isnan(m_) & ~np.isnan(m_err_) & ~np.isnan(time_)
m_ = m_[mask]
time_ = time_[mask]
m_err_ = m_err_[mask]

save_data = np.zeros((m_.size, 3))
save_data[:, 0] = time_
save_data[:, 1] = m_
save_data[:, 2] = m_err_
np.savetxt('Data/processed/lcmag_kepler.txt', save_data, header='Time\tMagnitude\tError', delimiter='\t')

# # # Save full lightcurve to file # # #
mask = ~np.isnan(m) & ~np.isnan(m_err) & ~np.isnan(time)
m = m[mask]
time = time[mask]
m_err = m_err[mask]

save_data = np.zeros((m.size, 3))
save_data[:, 0] = time
save_data[:, 1] = m
save_data[:, 2] = m_err
np.savetxt('Data/processed/lcmag_kepler_full.txt', save_data, header='Time\tMagnitude\tError', delimiter='\t')


# # # # # # # # PART 2 # # # # # # # # # #

# # # Fit polynomials close to eclipses # # #
eclipse1 = (phase > 0.1228) & (phase < 0.1515)
eclipse2 = (phase > 0.4620) & (phase < 0.4934)
mask_1 = ((phase > 0.0799) & (phase < 0.1976))
mask_2 = ((phase > 0.4283) & (phase < 0.5241))
mask_poly1 = mask_1 & ~eclipse1
mask_poly2 = mask_2 & ~eclipse2

poly1 = Polynomial.fit(phase[mask_poly1], flux_transit[mask_poly1], deg=2)
poly2 = Polynomial.fit(phase[mask_poly2], flux_transit[mask_poly2], deg=4)

time_1 = time[mask_1]
time_2 = time[mask_2]
phase_1 = phase[mask_1]
phase_2 = phase[mask_2]
flux_1 = flux_transit[mask_1] / poly1(phase_1)
flux_2 = flux_transit[mask_2] / poly2(phase_2)

if True:
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.scatter(phase_1, flux_transit[mask_1], 0.5, color='b', marker='.')
    # ax1.scatter(phase_1, flux_1, 0.5, color='k', marker='.')
    ax1.scatter(phase_1, poly1(phase_1), 0.3, color='r', marker='.')
    ax1.set_xlabel('Phase')
    ax2.scatter(phase_2, flux_transit[mask_2], 0.5, color='b', marker='.')
    # ax2.scatter(phase_2, flux_2, 0.5, color='k', marker='.')
    ax2.scatter(phase_2, poly2(phase_2), 0.3, color='r', marker='.')
    ax2.set_xlabel('Phase')
    plt.show()

# # Save flux # #
rmse_measure_mask = (phase_1 > 0.126) & (phase_1 < 0.149)
rmse_used_vals = flux_1[rmse_measure_mask]
mean_val = np.mean(rmse_used_vals)
error = np.sqrt(np.sum((mean_val - rmse_used_vals)**2) / rmse_used_vals.size)

flux = np.append(flux_1, flux_2)
time_12 = np.append(time_1, time_2)
flux_err = np.ones(flux.shape)
flux_err[:] = error

save_data = np.zeros((flux.size, 3))
save_data[:, 0] = time_12
save_data[:, 1] = flux
save_data[:, 2] = flux_err
np.savetxt('Data/processed/lcflux_kasoc_polyfit_full.txt', save_data, delimiter="\t")

# # Cut down data set to nearer eclipse # #
mask_1 = (phase_1 > 0.117) & (phase_1 < 0.156)
mask_2 = (phase_2 > 0.457) & (phase_2 < 0.500)

phase_1 = phase_1[mask_1]
phase_2 = phase_2[mask_2]
flux_1 = flux_1[mask_1]
flux_2 = flux_2[mask_2]
time_1 = time_1[mask_1]
time_2 = time_2[mask_2]


# # Convert to magnitudes and make error estimate # #
m_1 = -2.5*np.log10(flux_1)
m_2 = -2.5*np.log10(flux_2)
# Measure spread within full eclipse as estimator of flux error
rmse_measure_mask = (phase_1 > 0.126) & (phase_1 < 0.149)
rmse_used_vals = flux_1[rmse_measure_mask]
mean_val = np.mean(rmse_used_vals)
error = np.sqrt(np.sum((mean_val - rmse_used_vals)**2) / rmse_used_vals.size)
m_err_1 = np.abs(-2.5/np.log(10) * (error/flux_1))
m_err_2 = np.abs(-2.5/np.log(10) * (error/flux_2))

m = np.append(m_1, m_2)
m_err = np.append(m_err_1, m_err_2)
phase_12 = np.append(phase_1, phase_2)
time_12 = np.append(time_1, time_2)

mask = ~np.isnan(m) & ~np.isnan(m_err) & ~np.isnan(time_12) & ~np.isnan(phase_12)
m = m[mask]
time_12 = time_12[mask]
m_err = m_err[mask]
phase_12 = phase_12[mask]

if True:
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.errorbar(phase_1, m_1, m_err_1, fmt='k.', ecolor='gray', markersize=0.5, elinewidth=0.1, errorevery=10)
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Relative Magnitude')
    ax1.set_ylim([0.020, -0.003])
    ax2.errorbar(phase_2, m_2, m_err_2, fmt='k.', ecolor='gray', markersize=0.5, elinewidth=0.1, errorevery=10)
    ax2.set_xlabel('Phase')
    plt.show()

save_data = np.zeros((m.size, 3))
save_data[:, 0] = time_12
save_data[:, 1] = m
save_data[:, 2] = m_err
np.savetxt('Data/processed/lcmag_kepler_reduced.txt', save_data, delimiter='\t')


# # Save flux data # #
flux = np.append(flux_1, flux_2)
flux = flux[mask]
flux_err = np.ones(flux.shape)
flux_err[:] = error

if True:
    plt.figure()
    plt.errorbar(phase_12, flux, flux_err, fmt='k.', ecolor='gray', markersize=0.5, elinewidth=0.1, errorevery=10)
    plt.show()

save_data = np.zeros((flux.size, 3))
save_data[:, 0] = time_12
save_data[:, 1] = flux
save_data[:, 2] = flux_err
np.savetxt('Data/processed/lcflux_kasoc_reduced.txt', save_data, delimiter="\t")

