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

# Make design matrix and pass to linear regression corrector
dm = lk.DesignMatrix(tpf.flux[:, ~aper], name='regressors').pca(5).append_constant()
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
# Using pipeline aperture
aper = tpf_2min.pipeline_mask
raw_lc_2min = tpf_2min.to_lightcurve()

# Make design matrix
dm = lk.DesignMatrix(tpf_2min.flux[:, ~aper], name='pixels').pca(5).append_constant()

# Regression corrector object
reg = lk.RegressionCorrector(raw_lc_2min)
corrected_lc_2min = reg.correct(dm)

# Plot result
reg.diagnose()
plt.show(block=False)
print(raw_lc_2min.estimate_cdpp())
print(corrected_lc_2min.estimate_cdpp())

# # # RegressionCorrector on TESS 2-min cadence Target Pixel Files # # #
# Here regressioncorrector is done to remove trends due to e.g. spacecraft motion
tpf_2min = lk.search_targetpixelfile('KIC8430105', mission='TESS').download(quality_bitmask='hard')
# Using pipeline aperture
aper = tpf_2min.pipeline_mask
raw_lc_2min = tpf_2min.to_lightcurve()

# Make design matrix
dm = lk.DesignMatrix(tpf_2min.flux[:, ~aper], name='pixels').pca(4).append_constant()

# Regression corrector object
reg = lk.RegressionCorrector(raw_lc_2min)
corrected_lc_2min = reg.correct(dm)

# Plot result
reg.diagnose()
plt.show(block=False)

print(raw_lc_2min.estimate_cdpp())
print(corrected_lc_2min.estimate_cdpp())


# # # RegressionCorrector on TESS 2-min cadence Target Pixel Files # # #
# Here regressioncorrector is done to remove trends due to e.g. spacecraft motion
tpf_2min = lk.search_targetpixelfile('KIC8430105', mission='TESS').download(quality_bitmask='hard')
# Using pipeline aperture
aper = tpf_2min.pipeline_mask
raw_lc_2min = tpf_2min.to_lightcurve()

# Make design matrix
dm = lk.DesignMatrix(tpf_2min.flux[:, ~aper], name='pixels').pca(3).append_constant()

# Regression corrector object
reg = lk.RegressionCorrector(raw_lc_2min)
corrected_lc_2min = reg.correct(dm)

# Plot result
reg.diagnose()
plt.show(block=False)

print(raw_lc_2min.estimate_cdpp())
print(corrected_lc_2min.estimate_cdpp())


# # # RegressionCorrector on TESS 2-min cadence Target Pixel Files # # #
# Here regressioncorrector is done to remove trends due to e.g. spacecraft motion
tpf_2min = lk.search_targetpixelfile('KIC8430105', mission='TESS').download(quality_bitmask='hard')
# Using pipeline aperture
aper = tpf_2min.pipeline_mask
raw_lc_2min = tpf_2min.to_lightcurve()

# Make design matrix
dm = lk.DesignMatrix(tpf_2min.flux[:, ~aper], name='pixels').pca(2).append_constant()

# Regression corrector object
reg = lk.RegressionCorrector(raw_lc_2min)
corrected_lc_2min = reg.correct(dm)

# Plot result
#ax = raw_lc_2min.errorbar(label='Raw light curve')
#corrected_lc_2min.errorbar(ax=ax, label='Corrected light curve')
#plt.show(block=False)
reg.diagnose()
print(raw_lc_2min.estimate_cdpp())
print(corrected_lc_2min.estimate_cdpp())
plt.show(block=True)



#ax = corrected_lc_2min.normalize().plot(label='short cadence corrected lc')
# corrected_lc.normalize().plot(ax=ax, label='long cadence corrected lc')

#plt.show()

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
