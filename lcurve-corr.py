"""
lcurve-corr:
Scripts and functions for working with the incoming lightcurve from TESS (or Kepler) in order to prepare it for transit
photometry and asteroseismology.
"""

import lightkurve as lk
from lightkurve.collections import TargetPixelFileCollection, LightCurveCollection, LightCurveFileCollection
import numpy as np
import matplotlib.pyplot as plt
"""
pixelfile = lk.search_targetpixelfile('KIC8430105', mission='TESS').download_all(quality_bitmask='hard')
lc_file = lk.search_lightcurvefile('KIC8430105', mission='TESS').download_all(quality_bitmask='hard')
# pixelfile.plot(frame=-1)
# plt.show()
window_len = 801
if isinstance(pixelfile, TargetPixelFileCollection):
    lc = pixelfile[0].to_lightcurve(aperture_mask='pipeline').remove_nans().normalize() #.\
         # flatten(window_length=window_len)
    lc_thresh = pixelfile[0].to_lightcurve(aperture_mask='threshold').remove_nans() .normalize() # \
        # .flatten(window_length=window_len)
    for pxf in pixelfile:
        lc = lc.append(pxf.to_lightcurve(aperture_mask='pipeline').remove_nans().normalize())
                       # .flatten(window_length=window_len))
        lc_thresh = lc_thresh.append(pxf.to_lightcurve(aperture_mask='threshold').remove_nans().
                                     normalize()) # .flatten(window_length=window_len))
else:
    lc = pixelfile.to_lightcurve(aperture_mask='pipeline').remove_nans()  # .flatten(window_length=window_len)
    lc_thresh = pixelfile.to_lightcurve(aperture_mask='threshold').remove_nans()  # .flatten(window_length=window_len)

lc_sap = lc_file.SAP_FLUX
lc_pdcsap = lc_file.PDCSAP_FLUX


def custom_corrector_func(lc_):
    corrected_lc = lc_.normalize().flatten(window_length=801)
    return corrected_lc


if isinstance(lc_pdcsap, LightCurveCollection):
    lc_sap = lc_sap.stitch(corrector_func=custom_corrector_func).remove_nans()
    lc_pdcsap = lc_pdcsap.stitch(corrector_func=custom_corrector_func).remove_nans()


_, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
ax[0, 0].set_title('Aperture pipeline mask')
ax[0, 1].set_title('Aperture 3sigma threshold mask')
ax[1, 0].set_title('SAP lightcurve')
ax[1, 1].set_title('PDCSAP lightcurve')
lc.plot(ax=ax[0, 0])
lc_thresh.plot(ax=ax[0, 1])
lc_sap.plot(ax=ax[1, 0])
lc_pdcsap.plot(ax=ax[1, 1])

print("The CDPP noise metric equals:
{:.1f} ppm for aperture pipeline mask photometry;
{:.1f} ppm for aperture 3sigma threshold photometry;
{:.1f} ppm for SAP.".format(lc.estimate_cdpp(), lc_thresh.estimate_cdpp(), lc_sap.estimate_cdpp()))
plt.show()

lc_pdcsap.scatter()
plt.show()
"""

# # # TESSCut FFI Cutout work # # # FFI is the long cadence unprocessed data
# https://docs.lightkurve.org/tutorials/04-how-to-remove-tess-scattered-light-using-regressioncorrector.html
target = 'KIC8430105'

tpf = lk.search_tesscut(target).download_all(cutout_size=(30, 30), quality_bitmask='hard')
aper = tpf[0].create_threshold_mask()
# tpf[0].plot(aperture_mask=aper)
# plt.show(block=False)
raw_lc = tpf[0].to_lightcurve(aperture_mask=aper)
# raw_lc.plot()
# plt.show(block=False)

aper = tpf[1].create_threshold_mask()
tpf[1].plot(aperture_mask=aper)
plt.show(block=False)
raw_lc = tpf[1].to_lightcurve(aperture_mask=aper)
raw_lc.plot()
plt.show(block=False)
tpf = tpf[1]

# Detrend scattered light
regressors = tpf.flux[:, ~aper]
plt.figure()
plt.plot(regressors[:, :30])
plt.show(block=False)
dm = lk.DesignMatrix(regressors, name='regressors')
print(dm)
dm = dm.pca(5)
print(dm)
plt.figure()
plt.plot(tpf.time, dm.values + np.arange(5)*0.2, '.')
plt.show(block=False)

dm = dm.append_constant()
print(dm)
corrector = lk.RegressionCorrector(raw_lc)
corrected_lc = corrector.correct(dm)

corrector.diagnose()
plt.show(block=False)

# Check corrector model lightcurve for negative flux
model = corrector.model_lc
model.plot()
plt.show(block=False)
# Since it is below 0 in some cases, we correct this
model -= np.percentile(model.flux, 5)       # normalize to 5th percentile of model flux
model.plot()
plt.show(block=False)

ax = raw_lc.plot(label='Raw light curve')
corrected_lc = raw_lc - model
corrected_lc.plot(ax=ax, label='Corrected light curve')
plt.show()

