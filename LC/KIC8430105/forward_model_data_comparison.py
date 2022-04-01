import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams.update({'font.size': 17})

xlim1 = [-0.02110, 0.02181]
xlim2 = [0.63887, 0.67778]

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/')
lc_kepler = np.loadtxt('NOT/kepler_pdcsap/lc.out')
jktebop_model_deform = np.loadtxt('NOT/tess_forward/model.out')
lc_tess = np.loadtxt('../../LC/Data/processed/KIC8430105/lcmag_tess_tot.txt')

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

ax11.set_xlim(xlim1)
ax12.set_xlim(xlim2)
ylim = ax11.get_ylim()
ax11.set_ylim([ylim[1], ylim[0]])
ax12.set_ylim([ylim[1], ylim[0]])
ax12.set_yticks([])
ax11.set_xticks([-0.015, 0.0, 0.015])

phase_tess = np.mod(lc_tess[:, 0]-54998.2339005662, 63.3271055478)/63.3271055478
line_tess = ax2.plot(phase_tess, lc_tess[:, 1], '.', markersize=0.7)[0]
c_tess = line_tess.get_color()
xlim = ax2.get_xlim()
ax2.plot(phase_tess-1, lc_tess[:, 1], '.', markersize=0.7, color=c_tess)
ax2.plot(phase_tess+1, lc_tess[:, 1], '.', markersize=0.7, color=c_tess)
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

