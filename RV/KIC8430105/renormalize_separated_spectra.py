import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.modeling import models
from astropy import units as u

spectra = np.loadtxt('4500_5825_sep_flux.txt')
wavelength = spectra[:, 0]
flux_G = spectra[:, 1]      # giant
flux_M = spectra[:, 2]      # main sequence

T_estimate_G = 5042
T_estimate_M = 5708.73
R_G = 7.4791
R_M = 0.7502

flux_G = 1-flux_G
flux_M = 1-flux_M

bb_G = models.BlackBody(temperature=T_estimate_G*u.K)
bb_M = models.BlackBody(temperature=T_estimate_M*u.K)


def lum_ratio(wl, bb1, bb2, R1, R2):
    """
    Calculates luminosity ratio L2/L1 at specific wavelength wl
    """
    # return (R2/R1)**2 * (planck_func(wl, T2) / planck_func(wl, T1))
    return (R2/R1)**2 * (bb2(wl * u.AA).value / bb1(wl * u.AA).value)


L_M_o_G = lum_ratio(wavelength, bb_G, bb_M, R_G, R_M)
flux_G_norm = flux_G * (1+L_M_o_G) - L_M_o_G

L_G_o_M = lum_ratio(wavelength, bb_M, bb_G, R_M, R_G)
flux_M_norm = flux_M * (1+L_G_o_M) - L_G_o_M

plt.figure()
plt.plot(wavelength, flux_G_norm, 'r')
plt.plot(wavelength, flux_G, 'k')


plt.figure()
plt.plot(wavelength, flux_M_norm, 'r')
plt.plot(wavelength, flux_M, 'k')
plt.show()

# plt.plot(wavelength, L_M_o_G*L_G_o_M)
plt.plot(wavelength, L_M_o_G)
plt.xlabel('Wavelength (Å)')
plt.ylabel('L_MS / L_G')
plt.show()

save_array = np.empty(shape=spectra.shape)
save_array[:, 0] = wavelength
save_array[:, 1] = flux_G_norm
save_array[:, 2] = flux_M_norm
save_array[:, 3] = 1-spectra[:, 3]
save_array[:, 4] = 1-spectra[:, 4]
np.savetxt(
    'sep_flux_renormalized.txt', save_array, delimiter='\t',
    header='wavelength [Å]\t' + 'Separated flux Giant\t' + 'Separated flux MS\t' + 'Template Giant\t' + 'Template MS'
)
