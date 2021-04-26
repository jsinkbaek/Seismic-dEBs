import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation
import os
from scipy.interpolate import interp1d
from barycorrpy import get_BC_vel, get_stellar_data, utc_tdb
from matplotlib import pyplot as plt
import RV.spectrum_processing_functions as spf

# # # # Set variables for script # # # #
data_path = 'RV/Data/unprocessed/NOT/KIC8430105/'
data_out_path = 'RV/Data/processed/NOT/KIC8430105/'
masterfile_name = 'not_kic8430105.master.obs'
observatory_location = EarthLocation.of_site("lapalma")
observatory_name = "lapalma"
stellar_target = "kic8430105"
load_data = True       # Defines if normalized spectrum should be loaded from earlier, or done with AFS_algorithm

