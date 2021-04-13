import numpy as np
from astropy.io import fits
import os

loc_datafiles = 'Data/unprocessed/mast/kepler-kic8430105/'
crowding = np.array([])
quarter = np.array([])
tstart = np.array([])
tend = np.array([])
for filename in os.listdir(loc_datafiles):
    with fits.open(loc_datafiles+filename) as hdul:
        hdr0 = hdul[0].header
        hdr1 = hdul[1].header
        crowding = np.append(crowding, hdr1['CROWDSAP'])
        quarter = np.append(quarter, hdr0['QUARTER'])
        # tstart = np.append(tstart, hdr1['TSTART']+54833.0)
        # tend = np.append(tend, hdr1['TEND']+54833.0)
contamination = 1 - crowding
contamination_avg = np.average(contamination)
contamination_std = np.std(contamination)
print(contamination)
print(contamination_avg)
print(contamination_std)


