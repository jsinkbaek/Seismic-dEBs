import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


with fits.open("datafiles/kasoc/kplr008430105_kasoc-ts_llc_v1.fits") as hdul:
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

flux = (flux_seism * 1E-6 + 1) * corr_full
flux_transit = flux / (corr_long + corr_short)
period = 63.32713
phase = np.mod(time, period) / period


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


fig, axs = plt.subplots(2, 2, sharey='row')
ax1 = axs[0, 0]
ax2 = axs[0, 1]
ax3 = axs[1, 0]
ax4 = axs[1, 1]
ax1.plot(phase, flux/corr_long, 'r.', markersize=0.7)
ax1.legend(['LC/F_long'])
ax1.set_xlim([0.04, 0.24])
ax1.set_ylabel('Relative Flux')
ax2.plot(phase, flux/(corr_long+corr_short), 'b.', markersize=0.7)
ax2.legend(['LC/(F_long+F_short)'])
ax2.set_xlim([0.04, 0.24])
ax3.plot(phase, flux/corr_long, 'r.', markersize=0.7)
ax3.set_xlim([0.38, 0.58])
ax3.legend(['LC/F_long'])
ax3.set_ylabel('Relative Flux')
ax3.set_xlabel('Phase')
ax4.plot(phase, flux/(corr_long+corr_short), 'b.', markersize=0.7)
ax4.legend(['LC/(F_long+F_short)'])
ax4.set_xlim([0.38, 0.58])
ax4.set_xlabel('Phase')
plt.show(block=False)

plt.figure()
plt.plot(phase, corr_short, 'r.', markersize=1)
plt.plot(phase, corr_transit, 'b.', markersize=0.5)
plt.legend(['F_short', 'F_phase'])
plt.ylim([np.min(corr_short[~np.isnan(corr_short)]), np.max(corr_short[~np.isnan(corr_short)])])
plt.xlabel('Phase')
plt.ylabel('e-/s')
plt.show(block=False)
corr_long_plot = corr_long/np.median(corr_long)
plt.figure()
plt.plot(phase, corr_long_plot, 'r.', markersize=1)
plt.plot(phase, 1-corr_transit/np.min(corr_transit), 'b.', markersize=0.5)
plt.legend(['F_long', 'F_phase'])
plt.ylim([np.min(corr_long_plot[~np.isnan(corr_long_plot)]), np.max(corr_long_plot[~np.isnan(corr_long_plot)])])
plt.xlabel('Phase')
plt.ylabel('Normalized filter')
plt.show(block=True)

# # # Convert to magnitudes # # #
m = -2.5*np.log10(flux_transit)
m_err = np.abs(-2.5/np.log(10) * ((err_seism * 1E-6)*(corr_full/(corr_long+corr_short)))/flux_transit)
plt.figure()
plt.errorbar(time, m, m_err, fmt='k.', markersize=0.5, elinewidth=0.5)
plt.ylim([0.020, -0.003])
plt.show()

# # # Fold lightcurve and cut off data away from eclipses # # #
# plt.errorbar(phase, m, m_err, fmt='k.', markersize=0.5, elinewidth=0.5)
# plt.ylim([0.020, -0.003])
# plt.title('Select area to keep')
# coords = plt.ginput(n=4, timeout=0, show_clicks=True, mouse_add=1, mouse_stop=3, mouse_pop=2)
# x0, x1, x2, x3 = coords[0][0], coords[1][0], coords[2][0], coords[3][0]
x0=0.076
x1=0.20
x2=0.42
x3=0.53
print(x0, x1, x2, x3)
mask = ((phase>=x0) & (phase<=x1)) | ((phase>=x2) & (phase<=x3))

m_ = m[mask]
time_ = time[mask]
m_err_ = m_err[mask]
phase_ = phase[mask]

plt.errorbar(phase, m, m_err, fmt='.', color='gray', markersize=0.5, elinewidth=0.4)
plt.errorbar(phase_, m_, m_err_, fmt='k.', markersize=0.5, elinewidth=0.5)
plt.xlabel('Phase')
plt.ylabel('Relative Magnitude')
plt.legend(['Excluded data', 'Included data'])
plt.ylim([0.020, -0.003])
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
np.savetxt('lcmag_kepler.txt', save_data, header='Time\tMagnitude\tError', delimiter='\t')

# # # Save full lightcurve to file # # #
mask = ~np.isnan(m) & ~np.isnan(m_err) & ~np.isnan(time)
m = m[mask]
time = time[mask]
m_err = m_err[mask]

save_data = np.zeros((m.size, 3))
save_data[:, 0] = time
save_data[:, 1] = m
save_data[:, 2] = m_err
np.savetxt('lcmag_kepler_full.txt', save_data, header='Time\tMagnitude\tError', delimiter='\t')
