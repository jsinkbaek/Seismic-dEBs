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


# # # Save lightcurve to file # # #
save_data = np.zeros((m.size, 3))
save_data[:, 0] = time
save_data[:, 1] = m
save_data[:, 2] = m_err
np.savetxt('lcmag_kepler.txt', save_data, header='Time\tMagnitude\tError', delimiter='\t')