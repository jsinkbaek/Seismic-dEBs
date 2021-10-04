import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 17})

# Load and save uncorrected light curve # #
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
plt.figure(figsize=(16, 9))
flux = (flux_seism * 1E-6 + 1) * corr_full
tphase = np.mod(time, 120.5)
plt.plot(tphase, flux, 'r.', markersize=0.5)
plt.xlabel('BJD - 2400000', fontsize=22)
plt.ylabel('Flux [e/s]', fontsize=22)
plt.tight_layout()
plt.savefig('/home/sinkbaek/PycharmProjects/Seismic-dEBs/figures/report/kepler/raw.png', dpi=400)
plt.show()

save_data = np.empty((flux.size, 2))
save_data[:, 0] = time
save_data[:, 1] = flux
np.savetxt('Data/unprocessed/kic8430105_kepler_unfiltered.txt', save_data)

