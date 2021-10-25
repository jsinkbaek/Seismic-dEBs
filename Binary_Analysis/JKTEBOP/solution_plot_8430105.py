import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams.update({'font.size': 17})

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/')

lc_NOT_kepler = np.loadtxt('NOT/kepler_LTF/lc.out')
lc_NOT_kepler_2 = np.loadtxt('NOT/kepler_LTF (copy)/lc.out')
lc_gaulme_kepler = np.loadtxt('gaulme2016/KIC8430105/kepler_LTF/lc.out')
lc_NOT_tess = np.loadtxt('NOT/tess_LTF/lc.out')
lc_gaulme_tess = np.loadtxt('gaulme2016/KIC8430105/tess_ltf/lc.out')

rva_not = np.loadtxt('NOT/kepler_LTF/rvA.out')
rvb_not = np.loadtxt('NOT/kepler_LTF/rvB.out')
not_model = np.loadtxt('NOT/kepler_LTF/model.out')
gau_model = np.loadtxt('gaulme2016/KIC8430105/kepler_LTF/model.out')
rva_gau = np.loadtxt('gaulme2016/KIC8430105/kepler_LTF/rvA.out')
rvb_gau = np.loadtxt('gaulme2016/KIC8430105/kepler_LTF/rvB.out')
sys_gau_A = 16.1460728236
sys_gau_B = 16.6779564382
sys_not_A = 11.6126002555
sys_not_B = 12.0261221106


def lc_plot(lc_, xlim1, xlim2):
    mag = lc_[:, 1]
    err = lc_[:, 2]
    phase = lc_[:, 3]
    model = lc_[:, 4]
    omc = lc_[:, 5]
    sort_idx = np.argsort(phase)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(4, 2)
    ax11 = fig.add_subplot(gs[0:3, 0])
    ax12 = fig.add_subplot(gs[0:3, 1])
    ax2 = fig.add_subplot(gs[3, 0:2])
    ax21 = fig.add_subplot(gs[3, 0])
    ax22 = fig.add_subplot(gs[3, 1])

    ax11.errorbar(phase, mag, yerr=err, fmt='k.', ecolor='gray', markersize=2, elinewidth=0.7, errorevery=5)
    ax11.errorbar(phase-1, mag, yerr=err, fmt='k.', ecolor='gray', markersize=2, elinewidth=0.7, errorevery=5)
    ax12.errorbar(phase, mag, yerr=err, fmt='k.', ecolor='gray', markersize=2, elinewidth=0.7, errorevery=5)
    ax11.plot(phase[sort_idx], model[sort_idx], 'r-', linewidth=3.5)
    ax11.plot(phase[sort_idx]-1, model[sort_idx], 'r-', linewidth=3.5)
    ax12.plot(phase[sort_idx], model[sort_idx], 'r-', linewidth=3.5)

    ax11.set_xlim(xlim1)
    ax12.set_xlim(xlim2)
    ylim = ax11.get_ylim()
    ax11.set_ylim([ylim[1], ylim[0]])
    ax12.set_ylim([ylim[1], ylim[0]])

    ax21.errorbar(phase, omc, yerr=err, fmt='k.', ecolor='gray', markersize=2, elinewidth=0.7, errorevery=5)
    ax21.errorbar(phase-1, omc, yerr=err, fmt='k.', ecolor='gray', markersize=2, elinewidth=0.7, errorevery=5)
    ax22.errorbar(phase, omc, yerr=err, fmt='k.', ecolor='gray', markersize=2, elinewidth=0.7, errorevery=5)
    ax21.plot([-0.3, 0.5], [0.0, 0.0], '--', color='red', linewidth=3.5)
    ax22.plot([0.3, 1.0], [0.0, 0.0], '--', color='red', linewidth=3.5)
    ax21.set_xlim(xlim1)
    ax22.set_xlim(xlim2)
    ylim = ax21.get_ylim()
    ax21.set_ylim([ylim[1], ylim[0]])
    ax22.set_ylim([ylim[1], ylim[0]])

    ax11.set_ylabel('Relative Magnitude', fontsize=22)
    ax21.set_ylabel('O - C', fontsize=22)
    ax2.set_xlabel('Orbital Phase', fontsize=22)
    ax11.set_xticks([])
    ax12.set_xticks([])
    ax12.set_yticks([])
    ax22.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.setp(ax2.spines.values(), visible=False)
    ax2.xaxis.set_label_coords(0.5, -0.25)

    ax21.xaxis.set_major_locator(plt.MaxNLocator(4, symmetric=True, min_n_ticks=4))
    ax22.xaxis.set_major_locator(plt.MaxNLocator(4))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0)


def rv_plot(rv1_a, rv1_b, rv2_a, rv2_b, sys1_a, sys1_b, sys2_a, sys2_b, rv1_model, rv2_model):
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(5, 1)
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax2 = fig.add_subplot(gs[3:5, 0])
    ax21 = fig.add_subplot(gs[3, 0])
    ax22 = fig.add_subplot(gs[4, 0])

    ax22.set_xlabel('Orbital Phase', fontsize=25)
    ax1.set_ylabel('Radial Velocity [km/s]', fontsize=25)
    ax2.set_ylabel('O-C', fontsize=25)
    ax2.yaxis.set_label_coords(-0.05, 0.5)
    plt.setp(ax2.spines.values(), visible=False)
    ax1.set_xlim([-0.015, 1.0])
    ax21.set_xlim([-0.015, 1.0])
    ax22.set_xlim([-0.015, 1.0])

    ax1.errorbar(rv1_a[:, 3], rv1_a[:, 1]-sys1_a, yerr=rv1_a[:, 2], fmt='v', color='indianred', markersize=6)
    ax1.errorbar(rv1_a[:, 3]-1, rv1_a[:, 1]-sys1_a, yerr=rv1_a[:, 2], fmt='v', color='indianred', markersize=6)

    ax1.errorbar(rv1_b[:, 3], rv1_b[:, 1]-sys1_b, yerr=rv1_b[:, 2], fmt='v', color='royalblue', markersize=6)
    ax1.errorbar(rv1_b[:, 3]-1, rv1_b[:, 1]-sys1_b, yerr=rv1_b[:, 2], fmt='v', color='royalblue', markersize=6)

    ax1.errorbar(rv2_a[:, 3], rv2_a[:, 1]-sys2_a, yerr=rv2_a[:, 2], fmt='s', color='darkorange', markersize=6)
    ax1.errorbar(rv2_a[:, 3]-1, rv2_a[:, 1]-sys2_a, yerr=rv2_a[:, 2], fmt='s', color='darkorange', markersize=6)

    ax1.errorbar(rv2_b[:, 3], rv2_b[:, 1]-sys2_b, yerr=rv2_b[:, 2], fmt='s', color='blueviolet', markersize=6)
    ax1.errorbar(rv2_b[:, 3]-1, rv2_b[:, 1]-sys2_b, yerr=rv2_b[:, 2], fmt='s', color='blueviolet', markersize=6)

    ax1.plot(rv2_model[:, 0], rv2_model[:, 6]-sys2_a, '-.', alpha=0.8, color='darkorange')
    ax1.plot(rv2_model[:, 0]-1, rv2_model[:, 6]-sys2_a, '-.', alpha=0.8, color='darkorange')

    ax1.plot(rv2_model[:, 0], rv2_model[:, 7]-sys2_b, '-.', alpha=0.8, color='blueviolet')
    ax1.plot(rv2_model[:, 0]-1, rv2_model[:, 7]-sys2_b, '-.', alpha=0.8, color='blueviolet')

    ax1.plot(rv1_model[:, 0], rv1_model[:, 7] - sys1_b, linestyle='dotted', alpha=0.8, color='royalblue')
    ax1.plot(rv1_model[:, 0] - 1, rv1_model[:, 7] - sys1_b, linestyle='dotted', alpha=0.8, color='royalblue')
    ax1.plot(rv1_model[:, 0], rv1_model[:, 6] - sys1_a, linestyle='dotted', alpha=0.8, color='indianred')
    ax1.plot(rv1_model[:, 0] - 1, rv1_model[:, 6] - sys1_a, linestyle='dotted', alpha=0.8, color='indianred')

    ax1.plot([-0.05, 1.0], [0.0, 0.0], color='gray', alpha=0.6)

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax21.set_xticks([])

    # ax21.yaxis.set_ticks([-0.1, 0.1])

    ax21.errorbar(rv1_a[:, 3], rv1_a[:, 5], yerr=rv1_a[:, 2], fmt='v', color='indianred')
    ax21.errorbar(rv1_a[:, 3]-1, rv1_a[:, 5], yerr=rv1_a[:, 2], fmt='v', color='indianred')

    ax21.errorbar(rv2_a[:, 3], rv2_a[:, 5], yerr=rv2_a[:, 2], fmt='s', color='darkorange')
    ax21.errorbar(rv2_a[:, 3]-1, rv2_a[:, 5], yerr=rv2_a[:, 2], fmt='s', color='darkorange')

    ax21.plot([-0.05, 1], [0, 0], '--', color='black', alpha=0.7)
    ax22.errorbar(rv1_b[:, 3], rv1_b[:, 5], yerr=rv1_b[:, 2], fmt='v', color='royalblue')
    ax22.errorbar(rv1_b[:, 3]-1, rv1_b[:, 5], yerr=rv1_b[:, 2], fmt='v', color='royalblue')

    ax22.errorbar(rv2_b[:, 3], rv2_b[:, 5], yerr=rv2_b[:, 2], fmt='s', color='blueviolet')
    ax22.errorbar(rv2_b[:, 3]-1, rv2_b[:, 5], yerr=rv2_b[:, 2], fmt='s', color='blueviolet')

    ax22.plot([-0.05, 1], [0, 0], '--', color='black', alpha=0.7)

    std1_a = np.std(rv1_a[:, 5])
    std1_b = np.std(rv1_b[:, 5])
    std2_a = np.std(rv2_a[:, 5])
    std2_b = np.std(rv2_b[:, 5])

    ax21.fill_between([-0.05, 1], [std1_a, std1_a], [-std1_a, -std1_a], color='indianred', alpha=0.25)
    ax21.fill_between([-0.05, 1], [std2_a, std2_a], [-std2_a, -std2_a], color='darkorange', alpha=0.25)
    ax22.fill_between([-0.05, 1], [std1_b, std1_b], [-std1_b, -std1_b], color='royalblue', alpha=0.25)
    ax22.fill_between([-0.05, 1], [std2_b, std2_b], [-std2_b, -std2_b], color='blueviolet', alpha=0.25)

    ylim = ax1.get_ylim()
    ax1.set_ylim(ylim)
    ax1.plot([0.65892, 0.65892], [ylim[0], ylim[1]], linestyle='dotted', color='gray', alpha=0.5)
    ax1.plot([0., 0.], [ylim[0], ylim[1]], linestyle='dotted', color='gray', alpha=0.5)
    ylim = ax21.get_ylim()
    ax21.set_ylim(ylim)
    ax21.plot([0.65892, 0.65892], [ylim[0], ylim[1]], linestyle='dotted', color='gray', alpha=0.5)
    ax21.plot([0., 0.], [ylim[0], ylim[1]], linestyle='dotted', color='gray', alpha=0.5)
    ylim = ax22.get_ylim()
    ax22.set_ylim(ylim)
    ax22.plot([0.65892, 0.65892], [ylim[0], ylim[1]], linestyle='dotted', color='gray', alpha=0.5)
    ax22.plot([0., 0.], [ylim[0], ylim[1]], linestyle='dotted', color='gray', alpha=0.5)

    ax21.set_yticks([-2.5, 0.0, 2.5])
    ax22.set_yticks([-2.5, 0.0, 2.5])
    plt.tight_layout()


lc_plot(lc_NOT_tess, [-0.01756, 0.01859], [0.64334, 0.67481])
# plt.savefig('../../figures/report/tess/lc_not.png', dpi=400)
# lc_plot(lc_gaulme_kepler, [-0.02110, 0.02181], [0.63887, 0.67778])
# plt.savefig('../../figures/report/kepler/lc_gaulme.png', dpi=400)
lc_plot(lc_NOT_kepler, [-0.02110, 0.02181], [0.63887, 0.67778])
#plt.savefig('../../figures/report/kepler/lc_gaulme.png', dpi=400)
# rv_plot(rva_not, rvb_not, rva_gau, rvb_gau, 0, 0, 0, 0, not_model, gau_model)
# rv_plot(rva_not, rvb_not, rva_gau, rvb_gau, sys_not_A, sys_not_B, sys_gau_A, sys_gau_B, not_model, gau_model)
# plt.savefig('../../figures/report/kepler/rv_nosys.png', dpi=400)
plt.show()
