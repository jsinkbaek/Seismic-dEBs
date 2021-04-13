import numpy as np
from astropy.io import fits
import os

loc_datafiles = 'Data/unprocessed/mast/tess-kic8430105/'
crowding = np.array([])
for filename in os.listdir(loc_datafiles):
    with fits.open(loc_datafiles+filename) as hdul:
        hdr0 = hdul[0].header
        hdr1 = hdul[1].header
        crowding = np.append(crowding, hdr1['CROWDSAP'])

contamination = 1 - crowding
contamination_avg = np.average(contamination)
print(contamination)
print(contamination_avg)
