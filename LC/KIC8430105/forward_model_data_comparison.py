import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d

matplotlib.rcParams.update({'font.size': 15})

xlim1 = [-0.02110, 0.02181]
xlim2 = [0.63887, 0.67778]

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/')
lc_kepler = np.loadtxt('NOT/kepler_pdcsap/lc.out')
jktebop_model_deform = np.loadtxt('NOT/tess_forward/model.out')
lc_tess = np.loadtxt('../../LC/Data/processed/KIC8430105/lcmag_tess_tot.txt')
lc_flux_tess = np.loadtxt('../../LC/Data/processed/KIC8430105/lcflux_tess.txt')
phoebe_model = np.loadtxt('/home/sinkbaek/Dropbox/fwmodel_8430105.txt')
phoebe_tess = np.loadtxt('/home/sinkbaek/Dropbox/fwmodel_8430105_tess_short.txt')
print(np.median(np.diff(phoebe_model[:, 2]))*86400)
# phoebe_grid = np.linspace(np.min(phoebe_model[:, 2]), np.max(phoebe_model[:, 2]), phoebe_model[:, 2].size//4)
# intp_ph = interp1d(phoebe_model[:, 2], phoebe_model[:, 1], kind='linear')
# intp_ph_phase = interp1d(phoebe_model[:, 2], phoebe_model[:, 0])
# phoebe_model_new = np.empty((phoebe_grid.size, 3))
# phoebe_model_new[:, 0] = intp_ph_phase(phoebe_grid)
# phoebe_model_new[:, 1] = intp_ph(phoebe_grid)
# phoebe_model_new[:, 2] = phoebe_grid
# phoebe_model = phoebe_model_new
phase_phoebe = np.mod(phoebe_model[:, 2]-54998.2339005662, 63.3271055478)/63.3271055478
phase_phoebe_tess = np.mod(phoebe_tess[:, 2]-54998.2339005662, 63.3271055478)/63.3271055478
phase_jktebop = jktebop_model_deform[:, 0]



sort_idx_phoebe = np.argsort(phase_phoebe)
sort_idx_phtess = np.argsort(phase_phoebe_tess)


# Normalize phoebe forward model
fit_mask_prim = ((phase_phoebe > 0.95) & (phase_phoebe < 0.98363)) | ((phase_phoebe < 0.072) & (phase_phoebe > 0.01629))
fit_mask_sec = ((phase_phoebe > 0.5905) & (phase_phoebe < 0.6420)) | ((phase_phoebe < 0.7215) & (phase_phoebe > 0.6755))
fit_mask_prim_jk = ((phase_jktebop > 0.95) & (phase_jktebop < 0.98363)) | ((phase_jktebop < 0.072) & (phase_jktebop > 0.01629))
fit_mask_sec_jk = ((phase_jktebop > 0.5905) & (phase_jktebop < 0.6420)) | ((phase_jktebop < 0.7215) & (phase_jktebop > 0.6755))
norm_mask_prim = (phase_phoebe > 0.95) | (phase_phoebe < 0.072)
norm_mask_sec = (phase_phoebe > 0.5905) & (phase_phoebe < 0.7215)
fit_mask_prim_tess = ((phase_phoebe_tess > 0.95) & (phase_phoebe_tess < 0.98363)) | ((phase_phoebe_tess < 0.072) & (phase_phoebe_tess > 0.01629))
fit_mask_sec_tess = ((phase_phoebe_tess > 0.5905) & (phase_phoebe_tess < 0.6420)) | ((phase_phoebe_tess < 0.7215) & (phase_phoebe_tess > 0.6755))
norm_mask_prim_tess = (phase_phoebe_tess > 0.95) | (phase_phoebe_tess < 0.072)
norm_mask_sec_tess = (phase_phoebe_tess > 0.5905) & (phase_phoebe_tess < 0.7215)

phase_offset = -0.15
pfit_prim = Polynomial.fit(np.mod(phase_phoebe[fit_mask_prim]+phase_offset, 1.0), phoebe_model[fit_mask_prim, 1], deg=3)
pfit_sec = Polynomial.fit(phase_phoebe[fit_mask_sec], phoebe_model[fit_mask_sec, 1], deg=3)
pfit_prim_jk = Polynomial.fit(np.mod(phase_jktebop[fit_mask_prim_jk]+phase_offset, 1.0), jktebop_model_deform[fit_mask_prim_jk, 1], deg=3)
pfit_sec_jk = Polynomial.fit(phase_jktebop[fit_mask_sec_jk], jktebop_model_deform[fit_mask_sec_jk, 1], deg=3)
pfit_prim_tess = Polynomial.fit(np.mod(phase_phoebe_tess[fit_mask_prim_tess]+phase_offset, 1.0), phoebe_tess[fit_mask_prim_tess, 1], deg=3)
pfit_sec_tess = Polynomial.fit(phase_phoebe_tess[fit_mask_sec_tess], phoebe_tess[fit_mask_sec_tess, 1], deg=3)

ph_flux_norm_prim = phoebe_model[norm_mask_prim, 1] / pfit_prim(np.mod(phase_phoebe[norm_mask_prim]+phase_offset, 1.0))
ph_flux_norm_sec = phoebe_model[norm_mask_sec, 1] / pfit_sec(phase_phoebe[norm_mask_sec])
ph_phase_prim = phase_phoebe[norm_mask_prim]
ph_phase_sec = phase_phoebe[norm_mask_sec]
time_prim = phoebe_model[norm_mask_prim, 2]
time_sec = phoebe_model[norm_mask_sec, 2]
sort_prim = np.argsort(ph_phase_prim)
sort_sec = np.argsort(ph_phase_sec)

flux_offset_prim = pfit_prim(np.mod([phase_offset], 1.0))
flux_offset_sec = pfit_sec(0.6589472933)
flux_offset_ph = np.mean([flux_offset_prim, flux_offset_sec])
flux_offset_prim_tess = pfit_prim_tess(np.mod([phase_offset], 1.0))
flux_offset_sec_tess = pfit_sec_tess(0.6589472933)
flux_offset_ph_tess = np.mean([flux_offset_prim_tess, flux_offset_sec_tess])
flux_offset_jk_prim = pfit_prim_jk(np.mod([phase_offset], 1.0))
flux_offset_jk_sec = pfit_sec_jk(0.6589472933)
flux_offset_jk = np.mean([flux_offset_jk_sec, flux_offset_jk_prim])

p = plt.plot(phase_phoebe[sort_idx_phoebe], phoebe_model[sort_idx_phoebe, 1])[0]
c_ph = p.get_color()
plt.plot(phase_phoebe[sort_idx_phoebe]-1, phoebe_model[sort_idx_phoebe, 1], color=c_ph)
plt.plot(phase_phoebe[sort_idx_phoebe]+1, phoebe_model[sort_idx_phoebe, 1], color=c_ph)
plt.plot(phase_phoebe[norm_mask_prim], pfit_prim(np.mod(phase_phoebe[norm_mask_prim]+phase_offset, 1.0)), 'r.')
plt.plot(phase_phoebe[norm_mask_sec], pfit_sec(phase_phoebe[norm_mask_sec]), 'r.')
plt.xlim([-0.05, 1.05])
plt.show()

# Convert to magnitudes
phoebe_tess_scaled = np.copy(phoebe_tess)
phoebe_tess_mag = np.copy(phoebe_tess)
jktebop_model_deform[:, 1] = jktebop_model_deform[:, 1] - flux_offset_jk
phoebe_model[:, 1] = phoebe_model[:, 1] / flux_offset_ph
phoebe_model[:, 1] = -2.5*np.log10(phoebe_model[:, 1])
phoebe_tess_scaled[:, 1] = phoebe_tess[:, 1] + (1-flux_offset_ph_tess) # /flux_offset_ph_tess
phoebe_tess_mag[:, 1] = -2.5*np.log10(phoebe_tess_scaled[:, 1])
ph_mag_norm_prim = -2.5*np.log10(ph_flux_norm_prim)
ph_mag_norm_sec = -2.5*np.log10(ph_flux_norm_sec)

# Save
save_array = np.empty((ph_mag_norm_prim.size + ph_mag_norm_sec.size, 3))
save_array[0:ph_phase_prim.size, 0] = time_prim
save_array[ph_phase_prim.size:save_array[:, 0].size, 0] = time_sec
save_array[0:ph_phase_prim.size, 1] = ph_mag_norm_prim
save_array[ph_phase_prim.size:save_array[:, 0].size, 1] = ph_mag_norm_sec
save_array[:, 2] = 0.000007
np.savetxt('/home/sinkbaek/PycharmProjects/Seismic-dEBs/LC/Data/processed/KIC8430105/fwm_ph_norm.txt', save_array)

fig = plt.figure(figsize=(9, 9))
gs = fig.add_gridspec(2, 2)
ax = fig.add_subplot(gs[:, :])
ax11 = fig.add_subplot(gs[0, 0])
ax12 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, :])
# ax22 = fig.add_subplot(gs[1, 1])

ax11.plot(lc_kepler[:, 3], lc_kepler[:, 1], '.', color='grey', markersize=1)       # kepler normalized lightcurve
ax11.plot(lc_kepler[:, 3]-1, lc_kepler[:, 1], '.', color='grey', markersize=1, label='Norm. Kepler LC')
ax12.plot(lc_kepler[:, 3], lc_kepler[:, 1], '.', color='grey', markersize=1)
kepler_sort_idx = np.argsort(lc_kepler[:, 3])
ax11.plot(lc_kepler[kepler_sort_idx, 3], lc_kepler[kepler_sort_idx, 4], 'r-', linewidth=3)      # kepler norm model
ax11.plot(lc_kepler[kepler_sort_idx, 3]-1, lc_kepler[kepler_sort_idx, 4], 'r-', linewidth=3, label='JKTEBOP best fit')
ax12.plot(lc_kepler[kepler_sort_idx, 3], lc_kepler[kepler_sort_idx, 4], 'r-', linewidth=3)
line_phoebe = ax11.plot(ph_phase_prim[sort_prim], ph_mag_norm_prim[sort_prim], '--', linewidth=3)[0]
c_ph = line_phoebe.get_color()
ax11.plot(ph_phase_prim[sort_prim]-1, ph_mag_norm_prim[sort_prim], '--', linewidth=3, color=c_ph, label='Norm. PHOEBE model')
ax12.plot(ph_phase_sec[sort_sec], ph_mag_norm_sec[sort_sec], '--', linewidth=3, color=c_ph)
leg1 = ax11.legend()
leg1.legendHandles[0]._legmarker.set_markersize(12)

ax11.set_xlim(xlim1)
ax12.set_xlim(xlim2)
ylim = ax11.get_ylim()
ax11.set_ylim([0.025, -0.00222])
ax12.set_ylim([0.025, -0.00222])
ax12.set_yticks([])
ax11.set_xticks([-0.015, 0.0, 0.015])


phase_tess = np.mod(lc_tess[:, 0]-54998.2339005662, 63.3271055478)/63.3271055478
phase_flux_tess = np.mod(lc_flux_tess[:, 0]-54998.2339005662, 63.3271055478)/63.3271055478
#plt.figure()
#p_tess = plt.plot(phase_flux_tess, lc_flux_tess[:, 1], '.', markersize=0.7)[0]
#c_tess = p_tess.get_color()
#plt.plot(phase_flux_tess-1, lc_flux_tess[:, 1], '.', markersize=0.7, color=c_tess)
#plt.plot(phase_flux_tess+1, lc_flux_tess[:, 1], '.', markersize=0.7, color=c_tess)
#line_phoebe = plt.plot(phase_phoebe[sort_idx_phoebe], phoebe_tess[sort_idx_phoebe, 1])[0]
#line_phoebe_2 = plt.plot(phase_phoebe[sort_idx_phoebe], phoebe_tess_scaled[sort_idx_phoebe, 1])[0]
#c_phoebe = line_phoebe.get_color()
#c_phoebe_2 = line_phoebe_2.get_color()
#plt.plot(phase_phoebe[sort_idx_phoebe]-1, phoebe_tess[sort_idx_phoebe, 1], color=c_phoebe)
#plt.plot(phase_phoebe[sort_idx_phoebe]+1, phoebe_tess[sort_idx_phoebe, 1], color=c_phoebe)
#plt.plot(phase_phoebe[sort_idx_phoebe]-1, phoebe_tess_scaled[sort_idx_phoebe, 1], color=c_phoebe_2)
#plt.plot(phase_phoebe[sort_idx_phoebe]+1, phoebe_tess_scaled[sort_idx_phoebe, 1], color=c_phoebe_2)

# line_tess = ax2.plot(phase_tess, lc_tess[:, 1], '.', markersize=0.7, label='TESS light curve')[0]
# line_phoebe = ax2.plot(phase_phoebe_tess[sort_idx_phtess], phoebe_tess_mag[sort_idx_phtess, 1], label='PHOEBE forward model')[0]
line_phoebe = ax2.plot(phase_phoebe[sort_idx_phoebe], phoebe_model[sort_idx_phoebe, 1], '--', linewidth=3, label='PHOEBE forward model')[0]
c_phoebe = line_phoebe.get_color()
# c_tess = line_tess.get_color()
xlim = ax2.get_xlim()
# ax2.plot(phase_tess-1, lc_tess[:, 1], '.', markersize=0.7, color=c_tess)
# ax2.plot(phase_tess+1, lc_tess[:, 1], '.', markersize=0.7, color=c_tess)
# ax2.plot(phase_phoebe_tess[sort_idx_phtess]-1, phoebe_tess_mag[sort_idx_phtess, 1], color=c_phoebe)
# ax2.plot(phase_phoebe_tess[sort_idx_phtess]+1, phoebe_tess_mag[sort_idx_phtess, 1], color=c_phoebe)
# ax2.plot(jktebop_model_deform[:, 0], jktebop_model_deform[:, 1], 'r-', label='JKTEBOP forward model')
# ax2.plot(jktebop_model_deform[:, 0]-1, jktebop_model_deform[:, 1], 'r-')
# ax2.plot(jktebop_model_deform[:, 0]+1, jktebop_model_deform[:, 1], 'r-')
ax2.plot(phase_phoebe[sort_idx_phoebe]-1, phoebe_model[sort_idx_phoebe, 1], '--', linewidth=3, color=c_phoebe)
ax2.plot(phase_phoebe[sort_idx_phoebe]+1, phoebe_model[sort_idx_phoebe, 1], '--', linewidth=3, color=c_phoebe)
ax2.plot(jktebop_model_deform[:, 0], jktebop_model_deform[:, 1], 'r-', linewidth=3, label='JKTEBOP forward model')
ax2.plot(jktebop_model_deform[:, 0]-1, jktebop_model_deform[:, 1], 'r-', linewidth=3 )
ax2.plot(jktebop_model_deform[:, 0]+1, jktebop_model_deform[:, 1], 'r-', linewidth=3)
leg2 = ax2.legend(loc=(0.086, 0.01774))
leg2.legendHandles[0]._legmarker.set_markersize(12)

ax2.set_xlim(xlim)
ylim = ax2.get_ylim()
ax2.set_ylim([ylim[1], ylim[0]])
ax.set_xticks([])
ax.set_yticks([])
plt.setp(ax.spines.values(), visible=False)
ax.set_ylabel('Relative Magnitude', fontsize=22)
ax2.set_xlabel('Orbital phase', fontsize=22)
ax.yaxis.set_label_coords(-0.12, 0.5)

plt.subplots_adjust(wspace=0.06, hspace=0.15)
# plt.savefig('/home/sinkbaek/PycharmProjects/Seismic-dEBs/figures/report/forward.png', dpi=400)
plt.show()

fig = plt.figure(figsize=(9, 6))
gs = fig.add_gridspec(2, 2)
ax = fig.add_subplot(gs[:, :])
ax2 = fig.add_subplot(gs[:, :])

# ax11.set_xlim(xlim1)
# ax12.set_xlim(xlim2)
# ylim = ax11.get_ylim()
# ax11.set_ylim([0.025, -0.00222])
# ax12.set_ylim([0.025, -0.00222])
# ax12.set_yticks([])
# ax11.set_xticks([-0.015, 0.0, 0.015])

line_tess = ax2.plot(phase_tess, lc_tess[:, 1], '.', color='grey', markersize=0.6, label='TESS light curve')[0]
c_tess = line_tess.get_color()
ax2.plot(phase_tess-1, lc_tess[:, 1], '.', markersize=0.6, color=c_tess)
ax2.plot(phase_tess+1, lc_tess[:, 1], '.', markersize=0.6, color=c_tess)
ax2.plot(jktebop_model_deform[:, 0], jktebop_model_deform[:, 1], 'r-', label='JKTEBOP forward model', linewidth=4)
ax2.plot(jktebop_model_deform[:, 0]-1, jktebop_model_deform[:, 1], 'r-', linewidth=4)
ax2.plot(jktebop_model_deform[:, 0]+1, jktebop_model_deform[:, 1], 'r-', linewidth=4)
line_phoebe = ax2.plot(phase_phoebe_tess[sort_idx_phtess], phoebe_tess_mag[sort_idx_phtess, 1], '--', label='PHOEBE forward model', linewidth=4)[0]
# line_phoebe = ax2.plot(phase_phoebe[sort_idx_phoebe], phoebe_model[sort_idx_phoebe, 1], '--', linewidth=3, label='PHOEBE forward model')[0]
c_phoebe = line_phoebe.get_color()
xlim = ax2.get_xlim()
ax2.plot(phase_phoebe_tess[sort_idx_phtess]-1, phoebe_tess_mag[sort_idx_phtess, 1], '--', color=c_phoebe, linewidth=4)
ax2.plot(phase_phoebe_tess[sort_idx_phtess]+1, phoebe_tess_mag[sort_idx_phtess, 1], '--', color=c_phoebe, linewidth=4)

leg2 = ax2.legend(loc='lower center')
leg2.legendHandles[0]._legmarker.set_markersize(12)

ax2.set_xlim([0.6, 1.05])
ylim = ax2.get_ylim()
ax2.set_ylim([ylim[1], ylim[0]])
ax.set_xticks([])
ax.set_yticks([])
plt.setp(ax.spines.values(), visible=False)
ax.set_ylabel('Relative Magnitude', fontsize=22)
ax2.set_xlabel('Orbital phase', fontsize=22)
ax.yaxis.set_label_coords(-0.12, 0.5)

plt.subplots_adjust(wspace=0.06, hspace=0.15)
plt.savefig('/home/sinkbaek/PycharmProjects/Seismic-dEBs/figures/report/forward_tess.png', dpi=400)
plt.show()

