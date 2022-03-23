"""
tess-lcurve-corr:
Scripts and functions for working with the incoming lightcurve from TESS in order to prepare it for transit
photometry and asteroseismology. This script version only performs local trend fitting, no linear regression correction.
"""


import lightkurve as lk
from lightkurve.correctors import RegressionCorrector, DesignMatrix, PLDCorrector
# from lightkurve.collections import TargetPixelFileCollection, LightCurveCollection, LightCurveFileCollection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u, astropy.constants as c
import astropy as ap
from numpy.polynomial import Polynomial
import sys

# matplotlib.use('Qt5Agg')

# # Download 2-minute cadence targetpixelfiles and create light curve using pipeline aperture # #
target = 'KIC8430105'
print(lk.search_targetpixelfile(target, mission='TESS'))
tpf_2min = lk.search_targetpixelfile(target, mission='TESS').download(quality_bitmask='hard')
raw_lc = tpf_2min.to_lightcurve().remove_nans()
print(raw_lc.meta)
sys.exit()
raw_lc.plot()
plt.show(block=False)
lc_time = raw_lc.time.value


# # Select eclipses # #
eclipse1 = (lc_time > 1712.03) & (lc_time < 1713.88)
eclipse2 = (lc_time > 1733.52) & (lc_time < 1735.63)
eclipse = eclipse1 | eclipse2

flux_ecl = raw_lc.flux[eclipse]
time_ecl = lc_time[eclipse]
flux_necl = raw_lc.flux[~eclipse]
time_necl = lc_time[~eclipse]

plt.figure()
plt.plot(time_ecl, flux_ecl)
plt.show(block=False)


# # Fit polynomials # #
poly1_mask = (time_necl > 1711) & (time_necl < 1716.50)
poly2_mask = (time_necl > 1731.74) & (time_necl < 1737.54)
t_p1, f_p1 = time_necl[poly1_mask], flux_necl[poly1_mask]
t_p2, f_p2 = time_necl[poly2_mask], flux_necl[poly2_mask]

p1 = Polynomial.fit(t_p1, f_p1, deg=4)
p2 = Polynomial.fit(t_p2, f_p2, deg=3)

t_plot1 = np.linspace(t_p1[0], t_p1[-1], 1000)
t_plot2 = np.linspace(t_p2[0], t_p2[-1], 1000)

plt.figure()
plt.plot(time_necl, flux_necl)
plt.plot(t_plot1, p1(t_plot1), 'r--')
plt.plot(t_plot2, p2(t_plot2), 'r--')
plt.show(block=False)


# # cut down data and normalize with polynomials # #
t_dmask1 = (lc_time > 1711.95) & (lc_time < 1713.94)
t_dmask2 = (lc_time > 1733.42) & (lc_time < 1735.77)

tn1, tn2 = lc_time[t_dmask1], lc_time[t_dmask2]
fn1, fn2 = raw_lc.flux[t_dmask1].value/p1(tn1), raw_lc.flux[t_dmask2].value/p2(tn2)

tnorm, fnorm = np.append(tn1, tn2), np.append(fn1, fn2)
plt.figure()
plt.plot(tnorm, fnorm, 'r.', markersize=1)
plt.show(block=False)

# # Calculate noise as RMS in total eclipse # #
rmse_mask = (tn1 > 1712.188) & (tn1 < 1713.664)
rmse_vals = fn1[rmse_mask]
mean_val = np.mean(rmse_vals)
error = np.sqrt(np.sum((mean_val - rmse_vals)**2) / rmse_vals.size)

# # Convert to magnitudes # #
m = -2.5*np.log10(fnorm)
m_err = np.abs(-2.5/np.log(10) * (error/fnorm))

plt.figure()
plt.errorbar(tnorm, m, m_err, fmt='.', markersize=1, elinewidth=0.3)
plt.ylim([0.020, -0.003])
plt.xlim([1711.92, 1714.04])
plt.show(block=False)
plt.figure()
plt.errorbar(tnorm, m, m_err, fmt='.', markersize=1, elinewidth=0.3)
plt.ylim([0.020, -0.003])
plt.xlim([1733.40, 1735.76])
plt.show(block=False)

# # Save data to file # #
time_correct = 57000
save_data = np.zeros((m.size, 3))
save_data[:, 0] = tnorm + time_correct
save_data[:, 1] = m
save_data[:, 2] = m_err
np.savetxt('Data/processed/lcmag_tess_ltf.txt', save_data, delimiter='\t')

plt.show()