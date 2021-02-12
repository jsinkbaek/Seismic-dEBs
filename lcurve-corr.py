"""
lcurve-corr:
Scripts and functions for working with the incoming lightcurve from TESS (or Kepler) in order to prepare it for transit
photometry and asteroseismology.
"""

import lightkurve as lk
from lightkurve.correctors import KeplerCBVCorrector
from lightkurve.collections import TargetPixelFileCollection, LightCurveCollection, LightCurveFileCollection
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u, astropy.constants as c
import astropy as ap
from numpy.polynomial import Polynomial


target = 'KIC8430105'

"""
# # # TESSCut FFI Cutout work # # # FFI is the long cadence unprocessed data
# https://docs.lightkurve.org/tutorials/04-how-to-remove-tess-scattered-light-using-regressioncorrector.html

tpf = lk.search_tesscut(target).download_all(cutout_size=(30, 30), quality_bitmask='hard')
aper = tpf[0].create_threshold_mask()
# tpf[0].plot(aperture_mask=aper)
# plt.show(block=False)
raw_lc = tpf[0].to_lightcurve(aperture_mask=aper)
# raw_lc.plot()
# plt.show(block=False)

# Make aperture mask and raw light curve
aper = tpf[1].create_threshold_mask()
raw_lc = tpf[1].to_lightcurve(aperture_mask=aper)
tpf = tpf[1]
tpf.plot(aperture_mask=aper)
plt.show(block=False)


# Make design matrix and pass to linear regression corrector
dm = lk.DesignMatrix(tpf.flux[:, ~aper], name='regressors').pca(5)
plt.plot(tpf.time, dm.values + np.arange(5)*0.2, '.')
plt.show()
dm = dm.append_constant()
rc = lk.RegressionCorrector(raw_lc)
corrected_lc = rc.correct(dm)

# Remove the scattered light, allowing for large offset
corrected_lc = raw_lc - rc.model_lc + np.percentile(rc.model_lc.flux, 5)

# Plot results
ax = raw_lc.plot(label='Raw light curve')
corrected_lc.plot(ax=ax, label='Corrected light curve')
plt.show(block=False)
corrected_lc.plot()
plt.show(block=False)
rc.diagnose()
plt.show()
"""


# # # RegressionCorrector on TESS 2-min cadence Target Pixel Files # # #
# Here regressioncorrector is done to remove trends due to e.g. spacecraft motion
tpf_2min = lk.search_targetpixelfile('KIC8430105', mission='TESS').download(quality_bitmask='hard')
tpf_2min.plot(frame=500,  aperture_mask=tpf_2min.pipeline_mask, mask_color='red')
plt.show(block=False)
bckg_mask = np.ones(tpf_2min.shape[1:]) - tpf_2min.pipeline_mask
bckg_mask[-3:, -1:] = 0
bckg_mask[-2:, -2] = 0
bckg_mask[0:3, -2:] = 0
bckg_mask = bckg_mask > 0
tpf_2min.plot(frame=500,  aperture_mask=bckg_mask, mask_color='red')
plt.show()

# Using pipeline aperture
aper = tpf_2min.pipeline_mask
raw_lc_2min = tpf_2min.to_lightcurve()
# bckg_lc = tpf_2min.to_lightcurve(aperture_mask=bckg_mask)

# Make design matrix
dm = lk.DesignMatrix(tpf_2min.flux[:, ~aper], name='pixels').pca(2).append_constant()

# Regression corrector object
reg = lk.RegressionCorrector(raw_lc_2min)
# reg_bckg = lk.RegressionCorrector(bckg_lc)
corrected_lc_2min = reg.correct(dm)
# corrected_bckg = reg_bckg.correct(dm)

# Plot result
ax = raw_lc_2min.errorbar(label='Raw light curve')
corrected_lc_2min.errorbar(ax=ax, label='Corrected light curve')
plt.show(block=False)
reg.diagnose()
print(raw_lc_2min.estimate_cdpp())
print(corrected_lc_2min.estimate_cdpp())
plt.show(block=False)
plt.show()

# ax = corrected_lc_2min.normalize().plot(label='short cadence corrected lc')
# corrected_lc.normalize().plot(ax=ax, label='long cadence corrected lc')

# plt.show()

"""
# # # Kepler light curve detrending # # #
# https://docs.lightkurve.org/tutorials/04-removing-cbvs.html
cbvs = [1, 2, 3, 4, 5, 6, 7]
lcfs = lk.search_lightcurvefile(target, mission='Kepler').download_all(quality_bitmask='hard')
corr_lc = KeplerCBVCorrector(lcfs[0]).correct(cbvs=cbvs)
lcs = LightCurveCollection(corr_lc)
ax = corr_lc.normalize().plot(color='b', label='Corrected SAP')
lcfs[0].SAP_FLUX.normalize().plot(color='k', linestyle='dashed', ax=ax, label='SAP')
for lcf in lcfs[1:5]:
    corr_lc = KeplerCBVCorrector(lcf).correct(cbvs=cbvs)
    lcs.append(corr_lc)
    corr_lc.normalize().plot(color='b', ax=ax)
    lcf.SAP_FLUX.normalize().plot(color='k', linestyle='dashed', ax=ax)
plt.show()
"""

# # # Exclude transits # # #
lc = corrected_lc_2min.remove_nans()
# plt.plot(lc.flux)
# coords = plt.ginput(n=4, timeout=0, show_clicks=True, mouse_add=1, mouse_stop=3, mouse_pop=2)
# exclude = np.append(np.array(range(int(coords[0][0]), int(coords[1][0]))),
#                     np.array(range(int(coords[2][0]), int(coords[3][0]))))
# exclude = np.append(np.array(range(341, 1859)), np.array(range(14399, 16025)))
exclude1, exclude2 = np.array(range(486, 1763)), np.array(range(14495, 15929))
exclude = np.append(exclude1, exclude2)
# print(int(coords[0][0]), int(coords[1][0]), int(coords[2][0]), int(coords[3][0]))
mask = np.ones(lc.flux.shape, bool)
mask[exclude] = False
lc[mask].plot()
plt.show()

# # # Fit polynomials and normalize LC # # #
# plt.plot(lc[mask].flux, 'k--')
# plt.plot(exclude1[0], lc[mask][exclude1[0]].flux, 'r*')
# plt.plot(exclude2[0]-(exclude1[-1]-exclude1[0]), lc[mask][exclude2[0]-(exclude1[-1]-exclude1[0])].flux, 'r*')
# coords = plt.ginput(n=4, timeout=0, show_clicks=True, mouse_add=1, mouse_stop=3, mouse_pop=2)
# idx_fit1 = np.array(range(int(coords[0][0]), int(coords[1][0])))
# idx_fit2 = np.array(range(int(coords[2][0]), int(coords[3][0])))
# print(int(coords[0][0]), int(coords[1][0]), int(coords[2][0]), int(coords[3][0]))
# idx_fit1 = np.array(range(13, 5817))
idx_fit1 = np.array(range(13, 5017))
# idx_fit2 = np.array(range(8981, 13580))
idx_fit2 = np.array(range(8681, 13580))
lc_median = np.median(lc[mask].flux)
lc_fit1, lc_fit2 = lc[mask][idx_fit1]/lc_median, lc[mask][idx_fit2]/lc_median

p1 = Polynomial.fit(lc_fit1.time, lc_fit1.flux, deg=2)
p2 = Polynomial.fit(lc_fit2.time, lc_fit2.flux, deg=2)

ax = (lc/lc_median).plot(label='Full lc', color='g')
(lc[mask]/lc_median).plot(ax=ax, label='lc excluding transits')
lc_fit1.plot(ax=ax, label='lc for polynomial trend fit 1')
lc_fit2.plot(ax=ax, label='lc for polynomial trend fit 2')
plt.plot(lc_fit1.time, p1(lc_fit1.time), 'k--')
plt.plot(lc_fit2.time, p2(lc_fit2.time), 'k--')
plt.show()

idx_norm1 = np.array(range(idx_fit1[0], idx_fit1[-1]+(exclude1[-1]-exclude1[0])))
idx_norm2 = np.array(range(idx_fit2[0]+(exclude1[-1]-exclude1[0]), idx_fit2[-1]+(exclude1[-1]-exclude1[0])
                           + (exclude2[-1]-exclude2[0])))
lc_norm1, lc_norm2 = lc[idx_norm1]/lc_median, lc[idx_norm2]/lc_median
lc_norm1 = lc_norm1 / p1(lc_norm1.time)
lc_norm2 = lc_norm2 / p2(lc_norm2.time)
lc_norm = lc_norm1.append(lc_norm2)
# plt.plot(lc_norm.flux)
# coords = plt.ginput(n=4, timeout=0, show_clicks=True, mouse_add=1, mouse_stop=3, mouse_pop=2)
# print(int(coords[0][0]), int(coords[1][0]), int(coords[2][0]), int(coords[3][0]))
# include1 = np.array(range(int(coords[0][0]), int(coords[1][0])))
# include2 = np.array(range(int(coords[2][0]), int(coords[3][0])))
include1 = np.array(range(0, 2230))
include2 = np.array(range(10182, 12385))
include = np.append(include1, include2)
mask2 = np.zeros(lc_norm.flux.shape, bool)
mask2[include] = True
lc_norm = lc_norm[mask2]
lc_norm.plot()
plt.show()

# # # Convert to magnitudes # # #
m = -2.5*np.log(lc_norm.flux)
# https://en.wikipedia.org/wiki/Propagation_of_uncertainty
m_err = np.abs(-2.5/np.log(10) * lc_norm.flux_err/lc_norm.flux)

lc_mag = lc_norm
lc_mag.flux = m
lc_mag.flux_err = m_err
lc_mag.errorbar(ylabel='Relative magnitude')
plt.show()

# # # Save lightcurve to file # # #
time_correct = 57000
save_data = np.zeros((lc_mag.flux.size, 3))
save_data[:, 0] = lc_mag.time + time_correct
save_data[:, 1] = lc_mag.flux
save_data[:, 2] = lc_mag.flux_err
np.savetxt('lcmag.txt', save_data, header='Time\tMagnitude\tError', delimiter='\t')