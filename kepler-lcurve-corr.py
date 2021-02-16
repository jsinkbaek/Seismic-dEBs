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

median_value = np.median(corr_long)
flux_transit = (flux_seism + corr_full)/median_value - corr_long/median_value - corr_short/median_value + 1

plt.figure()
plt.plot(time, flux_seism+corr_full, 'r.', markersize=1)
plt.plot(time, corr_long, 'g.', markersize=0.5)
plt.show(block=False)

plt.figure()
plt.plot(time, corr_transit/median_value, 'r.', markersize=1)
plt.show(block=False)

plt.figure()
plt.plot(time, flux_transit + corr_short/median_value, 'r.', markersize=1)
plt.plot(time, flux_transit, 'b.', markersize=1)
plt.show(block=True)

# # # Convert to magnitudes # # #
m = -2.5*np.log10(flux_transit)
m_err = np.abs(-2.5/np.log(10) * (err_seism/median_value)/flux_transit)
plt.figure()
plt.errorbar(time, m, m_err, fmt='k.', markersize=0.5, elinewidth=0.5)
plt.ylim([0.020, -0.003])
plt.show()

# # # Fold lightcurve and cut off data away from eclipses # # #
period = 63.32713
phase = np.mod(time, period) / period
# plt.errorbar(phase, m, m_err, fmt='k.', markersize=0.5, elinewidth=0.5)
# plt.ylim([0.020, -0.003])
# plt.title('Select area to keep')
# coords = plt.ginput(n=4, timeout=0, show_clicks=True, mouse_add=1, mouse_stop=3, mouse_pop=2)
# x0, x1, x2, x3 = coords[0][0], coords[1][0], coords[2][0], coords[3][0]
x0=0.09563040975326542
x1=0.17768448464099507
x2=0.4452842604009781
x3=0.5169891726902555
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

# # # Save lightcurve to file # # #
mask = ~np.isnan(m_) & ~np.isnan(m_err_) & ~np.isnan(time_)
m_ = m_[mask]
time_ = time_[mask]
m_err_ = m_err_[mask]

save_data = np.zeros((m_.size, 3))
save_data[:, 0] = time_
save_data[:, 1] = m_
save_data[:, 2] = m_err_
np.savetxt('lcmag_kepler.txt', save_data, header='Time\tMagnitude\tError', delimiter='\t')
