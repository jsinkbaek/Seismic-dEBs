import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 18})

# Mass
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# not_kepler = np.array([1.2544743268, 0.8137981554, 0.0160041189, 0.0057547704])  # val A, val B, err A, err B
not_kepler = np.array([1.2544743268, 0.8137981554, 0.0157749, 0.0056894])  # val A, val B, err A, err B
not_tess = np.array([1.254074808, 0.8131272818, 0.0160455, 0.0057615])
# not_tess = np.array([1.254074808, 0.8131272818, 0.0113562015, 0.1715172969])
# gau_kepler = np.array([1.3235316213, 0.8310058351, 0.0299085038, 0.0177780483])
gau_kepler = np.array([1.3235316213, 0.8310058351, 0.0300639, 0.0178391])
gau_tess = np.array([1.323290508, 0.8309951783, 0.0176648076, 0.0119215196])

ax1.errorbar(1, not_kepler[0], yerr=not_kepler[2], fmt='rp', markersize=10)
ax1.errorbar(2, gau_kepler[0], yerr=gau_kepler[2], fmt='p', color='orange', markersize=10)
ax1.errorbar(3, not_tess[0], yerr=not_tess[2], fmt='bp', markersize=10)
ax1.errorbar(4, gau_tess[0], yerr=gau_tess[2], fmt='gp', markersize=10)
ax1.plot([0.7, 4.20], [1.52, 1.52], '--', color='black', linewidth=1.0)
ax1.fill_between([0.7, 4.20], [1.52-0.06, 1.52-0.06], [1.52+0.06, 1.52+0.06], alpha=0.4, color='grey')

ax1.set_xticks([1, 2, 3, 4])
ax1.set_xticklabels(['NOT Kepler', 'Gaulme Kepler', 'NOT TESS', 'Gaulme TESS'], rotation=20)
ax1.set_xlim([0.7, 4.2])

ax2.errorbar(1, not_kepler[1], yerr=not_kepler[3], fmt='rv', markersize=10)
ax2.errorbar(2, gau_kepler[1], yerr=gau_kepler[3], fmt='v', color='orange', markersize=10)
ax2.errorbar(3, not_tess[1], yerr=not_tess[3], fmt='bv', markersize=10)
ax2.errorbar(4, gau_tess[1], yerr=gau_tess[3], fmt='gv', markersize=10)

ax2.set_xticks([1, 2, 3, 4])
ax2.set_xticklabels(['NOT Kepler', 'Gaulme Kepler', 'NOT TESS', 'Gaulme TESS'], rotation=20)
ax1.set_ylabel(r'Giant mass in $M_{\bigodot}$', fontsize=22)
ax2.set_ylim([0.79, 0.87])

ax2.yaxis.tick_right()
ax2.set_ylabel(r'MS mass in $M_{\bigodot}$', fontsize=22, rotation=270)
ax2.yaxis.set_label_coords(1.125, 0.5)

plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0)

plt.savefig(fname='../figures/report/mass_radius/mass_2.png', dpi=400)


# Radius
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# not_kepler = np.array([7.532253316, 0.7539093043, 0.0421480452, 0.0054743409])
# not_tess = np.array([7.429325172, 0.7482196361, 0.0556111202, 0.0089069996])
# gau_kepler = np.array([7.687932365, 0.7708429083, 0.0652840906, 0.0074355405])
# gau_tess = np.array([7.546159157, 0.7603711045, 0.0528769941, 0.0081582503])
not_kepler = np.array([7.532253316, 0.7539093043, 0.0412123, 0.0053196])
not_tess = np.array([7.429325172, 0.7482196361, 0.0843203, 0.0113407])
gau_kepler = np.array([7.687932365, 0.7708429083, 0.0648540, 0.0074469])
gau_tess = np.array([7.546159157, 0.7603711045, 0.0528769941, 0.0081582503])

ax1.errorbar(1, not_kepler[0], yerr=not_kepler[2], fmt='rp', markersize=10)
ax1.errorbar(2, gau_kepler[0], yerr=gau_kepler[2], fmt='p', color='orange', markersize=10)
ax1.errorbar(3, not_tess[0], yerr=not_tess[2], fmt='bp', markersize=10)
ax1.errorbar(4, gau_tess[0], yerr=gau_tess[2], fmt='gp', markersize=10)
ax1.plot([0.80, 4.2], [7.65, 7.65], '--', color='black', linewidth=1.0)
ax1.fill_between([0.80, 4.20], [7.65-0.05, 7.65-0.05], [7.65+0.05, 7.65+0.05], alpha=0.4, color='grey')

ax1.set_xticks([1, 2, 3, 4])
ax1.set_xticklabels(['NOT Kepler', 'Gaulme Kepler', 'NOT TESS', 'Gaulme TESS'], rotation=20)
ax1.set_xlim([0.8, 4.2])

ax2.errorbar(1, not_kepler[1], yerr=not_kepler[3], fmt='rv', markersize=10)
ax2.errorbar(2, gau_kepler[1], yerr=gau_kepler[3], fmt='v', color='orange', markersize=10)
ax2.errorbar(3, not_tess[1], yerr=not_tess[3], fmt='bv', markersize=10)
ax2.errorbar(4, gau_tess[1], yerr=gau_tess[3], fmt='gv', markersize=10)

ax2.set_xticks([1, 2, 3, 4])
ax2.set_xticklabels(['NOT Kepler', 'Gaulme Kepler', 'NOT TESS', 'Gaulme TESS'], rotation=20)
ax1.set_ylabel(r'Giant radius in $R_{\bigodot}$', fontsize=22)

ax2.yaxis.tick_right()
ax2.set_ylabel(r'MS radius in $R_{\bigodot}$', fontsize=22, rotation=270)
ax2.yaxis.set_label_coords(1.125, 0.5)

plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig(fname='../figures/report/mass_radius/radius_2.png', dpi=400)

plt.show()
