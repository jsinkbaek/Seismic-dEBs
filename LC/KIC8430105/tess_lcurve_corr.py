"""
tess-lcurve-corr:
Scripts and functions for working with the incoming lightcurve from TESS in order to prepare it for transit
photometry and asteroseismology.
"""

import lightkurve as lk
from lightkurve.correctors import RegressionCorrector, DesignMatrix, PLDCorrector
# from lightkurve.collections import TargetPixelFileCollection, LightCurveCollection, LightCurveFileCollection
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u, astropy.constants as c
import astropy as ap
from numpy.polynomial import Polynomial
from copy import copy
import matplotlib


target = 'KIC8430105'
matplotlib.rcParams.update({'font.size': 22})


"""
# # # TESSCut FFI Cutout work # # # FFI (Full Frame Images) is the long cadence unprocessed data
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
fig = plt.figure(figsize=(11, 18))
gs = fig.add_gridspec(2, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
print(lk.search_targetpixelfile('KIC8430105', mission='TESS'))
tpf_2min = lk.search_targetpixelfile('KIC8430105', mission='TESS', sector=15).download(quality_bitmask='hard')
tpf_2min.plot(frame=300,  aperture_mask=tpf_2min.pipeline_mask, mask_color='red', ax=ax1)

# Using pipeline aperture or extended background mask
aper = tpf_2min.pipeline_mask
idx = np.where(aper)
aper2 = np.zeros(aper.shape, dtype='bool')
aper2[idx[0][0]-1:idx[0][-1]+1, np.min(idx[1])-1:np.max(idx[1])+1] = True
aper2[-2:, -2:] = True
aper2[-3:, -1] = True
aper2[0:3, -2:] = True
aper2[1:4, 2:5] = True
tpf_2min.plot(frame=300,  aperture_mask=~aper2, mask_color='red', ax=ax2)
ax1.set_xticks([])
ax2.set_xticks([])
ax1.set_yticks([])
ax2.set_yticks([])
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax2.set_yticklabels([])
ax1.set_xlabel(None)
ax2.set_xlabel(None)
ax1.set_ylabel(None)
ax2.set_ylabel(None)
ax1.set_title(None)
ax2.set_title(None)
plt.tight_layout()
# plt.savefig('/home/sinkbaek/PycharmProjects/Seismic-dEBs/figures/report/tess/apertures.png', dpi=400)
plt.show()
raw_lc_2min = tpf_2min.to_lightcurve()

# Make design matrix
dm = DesignMatrix(tpf_2min.flux[:, ~aper2], name='pixels').pca(2)
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot()
ax.plot(tpf_2min.time.value, dm.values + np.arange(2)*0.2, '.')
ax.set_yticks([])
ax.set_xlabel('Time - 2457000 [BTJD days]')
plt.tight_layout()
# plt.savefig('/home/sinkbaek/PycharmProjects/Seismic-dEBs/figures/report/tess/dm.png', dpi=400)
plt.show(block=True)
dm.validate()
dm = dm.append_constant()

# Regression corrector object
reg = RegressionCorrector(raw_lc_2min)
corrected_lc_2min = reg.correct(dm)

# Plot result
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot()
raw_lc_2min.errorbar(ax=ax, color='black')
# corrected_lc_2min.errorbar(ax=ax, label='Corrected light curve')
plt.tight_layout()
# plt.savefig('/home/sinkbaek/PycharmProjects/Seismic-dEBs/figures/report/tess/lc_raw.png', dpi=400)
plt.show(block=True)
reg.diagnose()
fig = plt.gcf()
axs = fig.axes
fig.set_size_inches(16, 9)
axs[0].xaxis.label.set_size(22)
axs[0].yaxis.label.set_size(22)
axs[1].xaxis.label.set_size(22)
axs[1].yaxis.label.set_size(22)
axs[0].tick_params(axis='both', labelsize=18)
axs[1].tick_params(axis='both', labelsize=18)
plt.tight_layout()
# plt.savefig('/home/sinkbaek/PycharmProjects/Seismic-dEBs/figures/report/tess/lc_corr.png', dpi=400)
print(raw_lc_2min.estimate_cdpp())
print(corrected_lc_2min.estimate_cdpp())
plt.show(block=True)


# # # Save unfitted light curve # # #
lc = corrected_lc_2min.remove_nans()
norm_mask = ((lc.time.value > 1713.9) & (lc.time.value < 1714.4)) | ((lc.time.value > 1732.9) & (lc.time.value < 1733.48))
norm_val = np.average(lc.flux[norm_mask])
lc_norm_tot = lc.flux / norm_val
lc_norm_err = lc.flux_err / norm_val
m_tot = -2.5*np.log10(lc_norm_tot)
# https://en.wikipedia.org/wiki/Propagation_of_uncertainty
m_err_tot = np.abs(-2.5/np.log(10) * lc_norm_err/lc_norm_tot)


time_correct = 57000
save_data = np.zeros((m_tot.size, 3))
save_data[:, 0] = lc.time.value + time_correct
save_data[:, 1] = m_tot
save_data[:, 2] = m_err_tot
# np.savetxt('Data/processed/lcmag_tess_tot.txt', save_data, delimiter='\t')

plt.figure()
plt.plot(lc.time.value, m_tot, 'r.')
plt.show()


# # # Exclude transits # # #

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

p1 = Polynomial.fit(lc_fit1.time.value, lc_fit1.flux, deg=2)
p2 = Polynomial.fit(lc_fit2.time.value, lc_fit2.flux, deg=2)

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot()
msize = matplotlib.rcParams['lines.markersize']
(lc/lc_median).scatter(ax=ax, label='Excluded from fits', color='gray', alpha=0.6, s=msize*1.5, marker='o')
(lc[mask]/lc_median).scatter(ax=ax, label='_no_legend_', color='gray', s=msize*1.5, marker='o', alpha=0.6)
lc_fit1.scatter(ax=ax, label='LC for polynomial trend fit 1', s=msize*1.5, marker='o')
lc_fit2.scatter(ax=ax, label='LC for polynomial trend fit 2', s=msize*1.5, marker='o')
plt.plot(lc_fit1.time.value, p1(lc_fit1.time.value), 'k--')
plt.plot(lc_fit2.time.value, p2(lc_fit2.time.value), 'k--')
ax.set_ylabel('Median Normalized Flux')
lgnd = ax.get_legend()
for handle in lgnd.legendHandles:
    handle.set_sizes([45])
ax.xaxis.label.set_size(22)
ax.yaxis.label.set_size(22)
ax.tick_params(axis='both', labelsize=18)
matplotlib.rcParams['legend.fontsize'] = 22
plt.tight_layout()
# plt.savefig('/home/sinkbaek/PycharmProjects/Seismic-dEBs/figures/report/tess/fit.png', dpi=400)
plt.show()

idx_norm1 = np.array(range(idx_fit1[0], idx_fit1[-1]+(exclude1[-1]-exclude1[0])))
idx_norm2 = np.array(range(idx_fit2[0]+(exclude1[-1]-exclude1[0]), idx_fit2[-1]+(exclude1[-1]-exclude1[0])
                           + (exclude2[-1]-exclude2[0])))
lc_norm1, lc_norm2 = lc[idx_norm1]/lc_median, lc[idx_norm2]/lc_median
lc_norm1 = lc_norm1 / p1(lc_norm1.time.value)
lc_norm2 = lc_norm2 / p2(lc_norm2.time.value)
lc_norm = lc_norm1.append(lc_norm2)
# plt.plot(lc_norm.flux)
# coords = plt.ginput(n=4, timeout=0, show_clicks=True, mouse_add=1, mouse_stop=3, mouse_pop=2)
# print(int(coords[0][0]), int(coords[1][0]), int(coords[2][0]), int(coords[3][0]))
# include1 = np.array(range(int(coords[0][0]), int(coords[1][0])))
# include2 = np.array(range(int(coords[2][0]), int(coords[3][0])))
include1 = np.array(range(360, 1870))
include2 = np.array(range(10682, 12385))
include = np.append(include1, include2)
mask2 = np.zeros(lc_norm.flux.shape, bool)
mask2[include] = True
lc_norm = lc_norm[mask2]
lc_norm.plot()
plt.show()

# # # Convert to magnitudes # # #
m = -2.5*np.log10(lc_norm.flux)
# https://en.wikipedia.org/wiki/Propagation_of_uncertainty
m_err = np.abs(-2.5/np.log(10) * lc_norm.flux_err/lc_norm.flux)

lc_mag = copy(lc_norm)
lc_mag.flux = m
lc_mag.flux_err = m_err
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 2)
ax3 = fig.add_subplot(gs[0, :])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

lc_mag.errorbar(ax=ax1, ylabel='Relative magnitude', xlabel='', label='_nolegend_', fmt='k.', ecolor='gray', errorevery=5, markersize=2, elinewidth=0.5)
lc_mag.errorbar(ax=ax2, ylabel='', xlabel='', fmt='k.', label='_nolegend_', ecolor='gray', errorevery=5, markersize=2, elinewidth=0.5)
ax1.set_ylim([0.022, -0.006])
ax2.set_xlim([1711.9, 1714])
ax2.set_ylim([0.022, -0.006])
ax1.set_xlim([1733.3, 1735.715])
ax2.set_yticklabels([])
ax3.set_xlabel('Time - 2457000 [BTJD days]')
ax3.xaxis.set_label_coords(0.5, -0.10)
plt.setp(ax3.spines.values(), visible=False)
# ax1.xaxis.set_major_locator(plt.MaxNLocator(4, min_n_ticks=4))
# ax2.xaxis.set_major_locator(plt.MaxNLocator(4, min_n_ticks=4))
ax2.set_xticks([1712.2, 1712.7, 1713.2, 1713.7])
ax3.set_xticks([])
ax3.set_yticks([])
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig('/home/sinkbaek/PycharmProjects/Seismic-dEBs/figures/report/tess/truncated.png', dpi=400)
plt.show(block=True)

# # # Save lightcurve to file # # #
time_correct = 57000
save_data = np.zeros((lc_mag.flux.size, 3))
save_data[:, 0] = lc_mag.time.value + time_correct
save_data[:, 1] = lc_mag.flux
save_data[:, 2] = lc_mag.flux_err
np.savetxt('Data/processed/lcmag_tess.txt', save_data, delimiter='\t')
save_data[:, 0] = lc_norm.time.value + time_correct
save_data[:, 1] = lc_norm.flux
save_data[:, 2] = lc_norm.flux_err
np.savetxt('Data/processed/lcflux_tess.txt', save_data, delimiter='\t')
