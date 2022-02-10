import numpy as np
import matplotlib.pyplot as plt
import os

spectra = np.loadtxt('4500_5825_sep_flux.txt')
wavelength = spectra[:, 0]
flux_G = spectra[:, 1]      # giant
flux_M = spectra[:, 2]      # main sequence


