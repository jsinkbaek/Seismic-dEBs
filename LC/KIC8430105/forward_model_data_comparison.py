import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from numpy.polynomial import Polynomial

matplotlib.rcParams.update({'font.size': 17})

xlim1 = [-0.02110, 0.02181]
xlim2 = [0.63887, 0.67778]

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/')
lc_kepler = np.loadtxt('NOT/kepler_pdcsap/lc.out')
jktebop_model_deform = np.loadtxt('NOT/tess_forward/model.out')
lc_tess = np.loadtxt('../../LC/Data/processed/KIC8430105/lcmag_tess_tot.txt')
phoebe_model = np.loadtxt('/home/sinkbaek/Dropbox/fwmodel_8430105.txt')
phase_phoebe = np.mod(phoebe_model[:, 2]-54998.2339005662, 63.3271055478)/63.3271055478
sort_idx_phoebe = np.argsort(phase_phoebe)

# Normalize phoebe forward model
fit_mask_prim = ((phase_phoebe > 0.95) & (phase_phoebe < 0.98363)) | ((phase_phoebe < 0.072) & (phase_phoebe > 0.01629))
fit_mask_sec = ((phase_phoebe > 0.5905) & (phase_phoebe < 0.6420)) | ((phase_phoebe < 0.7215) & (phase_phoebe > 0.6755))
norm_mask_prim = (phase_phoebe > 0.95) | (phase_phoebe < 0.072)
norm_mask_sec = (phase_phoebe > 0.5905) & (phase_phoebe < 0.7215)

phase_offset = -0.15
pfit_prim = Polynomial.fit(np.mod(phase_phoebe[fit_mask_prim]+phase_offset, 1.0), phoebe_model[fit_mask_prim, 1], deg=3)
pfit_sec = Polynomial.fit(phase_phoebe[fit_mask_sec], phoebe_model[fit_mask_sec, 1], deg=3)

ph_flux_norm_prim = phoebe_model[norm_mask_prim, 1] / pfit_prim(np.mod(phase_phoebe[norm_mask_prim]+phase_offset, 1.0))
ph_flux_norm_sec = phoebe_model[norm_mask_sec, 1] / pfit_sec(phase_phoebe[norm_mask_sec])
ph_phase_prim = phase_phoebe[norm_mask_prim]
ph_phase_sec = phase_phoebe[norm_mask_sec]
sort_prim = np.argsort(ph_phase_prim)
sort_sec = np.argsort(ph_phase_sec)

flux_offset_prim = pfit_prim(np.mod([phase_offset], 1.0))
flux_offset_sec = pfit_sec(0.6589472933)
flux_offset_ph = np.mean([flux_offset_prim, flux_offset_sec])

p = plt.plot(phase_phoebe[sort_idx_phoebe], phoebe_model[sort_idx_phoebe, 1])[0]
c_ph = p.get_color()
plt.plot(phase_phoebe[sort_idx_phoebe]-1, phoebe_model[sort_idx_phoebe, 1], color=c_ph)
plt.plot(phase_phoebe[sort_idx_phoebe]+1, phoebe_model[sort_idx_phoebe, 1], color=c_ph)
plt.plot(phase_phoebe[norm_mask_prim], pfit_prim(np.mod(phase_phoebe[norm_mask_prim]+phase_offset, 1.0)), 'r.')
plt.plot(phase_phoebe[norm_mask_sec], pfit_sec(phase_phoebe[norm_mask_sec]), 'r.')
plt.xlim([-0.05, 1.05])
plt.show()

# Convert to magnitudes
phoebe_model[:, 1] = phoebe_model[:, 1] / flux_offset_ph
phoebe_model[:, 1] = -2.5*np.log10(phoebe_model[:, 1])
ph_mag_norm_prim = -2.5*np.log10(ph_flux_norm_prim)
ph_mag_norm_sec = -2.5*np.log10(ph_flux_norm_sec)

fig = plt.figure(figsize=(9, 9))
gs = fig.add_gridspec(2, 2)
ax = fig.add_subplot(gs[:, :])
ax11 = fig.add_subplot(gs[0, 0])
ax12 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, :])
# ax22 = fig.add_subplot(gs[1, 1])

ax11.plot(lc_kepler[:, 3], lc_kepler[:, 1], 'k.', markersize=0.7)       # kepler normalized lightcurve
ax11.plot(lc_kepler[:, 3]-1, lc_kepler[:, 1], 'k.', markersize=0.7)
ax12.plot(lc_kepler[:, 3], lc_kepler[:, 1], 'k.', markersize=0.7)
kepler_sort_idx = np.argsort(lc_kepler[:, 3])
ax11.plot(lc_kepler[kepler_sort_idx, 3], lc_kepler[kepler_sort_idx, 4], 'r-', linewidth=2)      # kepler norm model
ax11.plot(lc_kepler[kepler_sort_idx, 3]-1, lc_kepler[kepler_sort_idx, 4], 'r-', linewidth=2)
ax12.plot(lc_kepler[kepler_sort_idx, 3], lc_kepler[kepler_sort_idx, 4], 'r-', linewidth=2)
line_phoebe = ax11.plot(ph_phase_prim[sort_prim], ph_mag_norm_prim[sort_prim])[0]
c_ph = line_phoebe.get_color()
ax11.plot(ph_phase_prim[sort_prim]-1, ph_mag_norm_prim[sort_prim], color=c_ph)
ax12.plot(ph_phase_sec[sort_sec], ph_mag_norm_sec[sort_sec], color=c_ph)

ax11.set_xlim(xlim1)
ax12.set_xlim(xlim2)
ylim = ax11.get_ylim()
ax11.set_ylim([ylim[1], ylim[0]])
ax12.set_ylim([ylim[1], ylim[0]])
ax12.set_yticks([])
ax11.set_xticks([-0.015, 0.0, 0.015])

phase_tess = np.mod(lc_tess[:, 0]-54998.2339005662, 63.3271055478)/63.3271055478
line_tess = ax2.plot(phase_tess, lc_tess[:, 1], '.', markersize=0.7)[0]
line_phoebe = ax2.plot(phase_phoebe[sort_idx_phoebe], phoebe_model[sort_idx_phoebe, 1])[0]
c_phoebe = line_phoebe.get_color()
c_tess = line_tess.get_color()
xlim = ax2.get_xlim()
ax2.plot(phase_tess-1, lc_tess[:, 1], '.', markersize=0.7, color=c_tess)
ax2.plot(phase_tess+1, lc_tess[:, 1], '.', markersize=0.7, color=c_tess)
ax2.plot(phase_phoebe[sort_idx_phoebe]-1, phoebe_model[sort_idx_phoebe, 1], color=c_phoebe)
ax2.plot(phase_phoebe[sort_idx_phoebe]+1, phoebe_model[sort_idx_phoebe, 1], color=c_phoebe)
ax2.plot(jktebop_model_deform[:, 0], jktebop_model_deform[:, 1], 'r-')
ax2.plot(jktebop_model_deform[:, 0]-1, jktebop_model_deform[:, 1], 'r-')
ax2.plot(jktebop_model_deform[:, 0]+1, jktebop_model_deform[:, 1], 'r-')

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

plt.show()

