import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 18})

# # # Mass # # #
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

not_kepler = np.array([1.2544743268, 0.8137981554, 0.01453, 0.00521])  # val A, val B, err A, err B
not_tess = np.array([1.254074808, 0.8131272818, 0.01451, 0.00522])

gau_kepler = np.array([1.3235316213, 0.8310058351, 0.02908, 0.01724])
gau_tess = np.array([1.323290508, 0.8309951783, 0.0291881, 0.01726208])

gau_16_m = np.array([1.31, 0.83, 0.02, 0.01])

markersize = 12
ax1.errorbar(1, not_kepler[0], yerr=not_kepler[2], fmt='rp', markersize=markersize)
ax1.errorbar(2, gau_kepler[0], yerr=gau_kepler[2], fmt='p', color='orange', markersize=markersize)
ax1.errorbar(3, not_tess[0], yerr=not_tess[2], fmt='bp', markersize=markersize)
ax1.errorbar(4, gau_tess[0], yerr=gau_tess[2], fmt='gp', markersize=markersize)
ax1.errorbar(5, gau_16_m[0], yerr=gau_16_m[2], fmt='p', color='indigo', markersize=markersize)
ax1.plot([0.7, 5.20], [1.52, 1.52], '--', color='black', linewidth=1.0)
ax1.plot([0.7, 5.20], [1.518, 1.518], '.-', color='red', linewidth=1.0)
ax1.fill_between([0.7, 5.20], [1.52-0.06, 1.52-0.06], [1.52+0.06, 1.52+0.06], alpha=0.4, color='grey')
ax1.fill_between([0.7, 5.20], [1.518-0.082, 1.518-0.082], [1.518+0.082, 1.518+0.082], alpha=0.4, color='indianred')

ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_xticklabels(['NOT Kepler', 'Gaulme Kepler', 'NOT TESS', 'Gaulme TESS', 'Gaulme et al. 2016'], rotation=20)
ax1.set_xlim([0.7, 5.2])

ax2.errorbar(1, not_kepler[1], yerr=not_kepler[3], fmt='rv', markersize=markersize)
ax2.errorbar(2, gau_kepler[1], yerr=gau_kepler[3], fmt='v', color='orange', markersize=markersize)
ax2.errorbar(3, not_tess[1], yerr=not_tess[3], fmt='bv', markersize=markersize)
ax2.errorbar(4, gau_tess[1], yerr=gau_tess[3], fmt='gv', markersize=markersize)
ax2.errorbar(5, gau_16_m[1], yerr=gau_16_m[3], fmt='v', color='indigo', markersize=markersize)

ax2.set_xticks([1, 2, 3, 4, 5])
ax2.set_xticklabels(['NOT Kepler', 'Gaulme Kepler', 'NOT TESS', 'Gaulme TESS', 'Gaulme et al. 2016'], rotation=20)
ax1.set_ylabel(r'Giant component mass [$M_{\bigodot}$]', fontsize=22)
ax2.set_ylim([0.79, 0.87])

ax2.yaxis.tick_right()
ax2.set_ylabel(r'Main sequence component mass [$M_{\bigodot}$]', fontsize=22, rotation=270)
ax2.yaxis.set_label_coords(1.125, 0.5)

plt.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0)

plt.savefig(fname='../figures/report/mass_radius/mass.png', dpi=400)


# # # Radius # # #
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

not_kepler = np.array([7.532253316, 0.7539093043, 0.04258, 0.00537])
not_tess = np.array([7.429325172, 0.7482196361, 0.08126, 0.01148])
gau_kepler = np.array([7.687932365, 0.7708429083, 0.0656182, 0.007403574])
gau_tess = np.array([7.546159157, 0.7603711045, 0.123442, 0.01633766])

gau_16_r = np.array([7.65, 0.770, 0.05, 0.005])

ax1.errorbar(1, not_kepler[0], yerr=not_kepler[2], fmt='rp', markersize=markersize)
ax1.errorbar(2, gau_kepler[0], yerr=gau_kepler[2], fmt='p', color='orange', markersize=markersize)
ax1.errorbar(3, not_tess[0], yerr=not_tess[2], fmt='bp', markersize=markersize)
ax1.errorbar(4, gau_tess[0], yerr=gau_tess[2], fmt='gp', markersize=markersize)
ax1.errorbar(5, gau_16_r[0], yerr=gau_16_r[2], fmt='p', color='indigo', markersize=markersize)
ax1.plot([0.80, 5.2], [8.1, 8.1], '--', color='black', linewidth=1.0)
ax1.plot([0.80, 5.2], [8.086, 8.086], '.-', color='red', linewidth=1.0)
ax1.fill_between([0.80, 5.20], [8.1-0.1, 8.1-0.1], [8.1+0.1, 8.1+0.1], alpha=0.4, color='grey')
ax1.fill_between([0.80, 5.20], [8.086-0.1655, 8.086-0.1655], [8.086+0.1655, 8.086+0.1655], alpha=0.4, color='indianred')

ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_xticklabels(['NOT Kepler', 'Gaulme Kepler', 'NOT TESS', 'Gaulme TESS', 'Gaulme et al. 2016'], rotation=20)
ax1.set_xlim([0.8, 5.2])

ax2.errorbar(1, not_kepler[1], yerr=not_kepler[3], fmt='rv', markersize=markersize)
ax2.errorbar(2, gau_kepler[1], yerr=gau_kepler[3], fmt='v', color='orange', markersize=markersize)
ax2.errorbar(3, not_tess[1], yerr=not_tess[3], fmt='bv', markersize=markersize)
ax2.errorbar(4, gau_tess[1], yerr=gau_tess[3], fmt='gv', markersize=markersize)
ax2.errorbar(5, gau_16_r[1], yerr=gau_16_r[3], fmt='v', color='indigo', markersize=markersize)

ax2.set_xticks([1, 2, 3, 4, 5])
ax2.set_xticklabels(['NOT Kepler', 'Gaulme Kepler', 'NOT TESS', 'Gaulme TESS', 'Gaulme et al. 2016'], rotation=20)
ax1.set_ylabel(r'Giant component radius [$R_{\bigodot}$]', fontsize=22)

ax2.yaxis.tick_right()
ax2.set_ylabel(r'Main sequence component radius [$R_{\bigodot}$]', fontsize=22, rotation=270)
ax2.yaxis.set_label_coords(1.125, 0.5)

plt.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0)
plt.savefig(fname='../figures/report/mass_radius/radius.png', dpi=400)

# # # log g # # #

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])

logg_not_kepler = np.array([2.7828, 4.5946, 0.00456, 0.00545])
logg_not_tess = np.array([2.7945, 4.600, 0.00919, 0.01270])
logg_gau_kepler = np.array([2.7882, 4.5837, 0.0058906, 0.00740033])
logg_gau_tess = np.array([2.804, 4.596, 0.0132532, 0.01787753])

logg_gau_16 = np.array([2.788, 4.5, 0.004, 4.5])

ax1.errorbar(1, logg_not_kepler[0], yerr=logg_not_kepler[2], fmt='rp', markersize=markersize)
ax1.errorbar(2, logg_gau_kepler[0], yerr=logg_gau_kepler[2], fmt='p', color='orange', markersize=markersize)
ax1.errorbar(3, logg_not_tess[0], yerr=logg_not_tess[2], fmt='bp', markersize=markersize)
ax1.errorbar(4, logg_gau_tess[0], yerr=logg_gau_tess[2], fmt='gp', markersize=markersize)
ax1.errorbar(5, logg_gau_16[0], yerr=logg_gau_16[2], fmt='p', color='indigo', markersize=markersize)
ax1.plot([0.80, 5.2], [2.802, 2.802], '--', color='black', linewidth=1.0)
ax1.plot([0.80, 5.2], [2.8034, 2.8034], '.-', color='red', linewidth=1.0)
ax1.fill_between([0.80, 5.20], [2.802-0.004, 2.802-0.004], [2.802+0.004, 2.802+0.004], alpha=0.4, color='grey')
ax1.fill_between([0.80, 5.20], [2.8034-0.0069, 2.8034-0.0069], [2.8034+0.0069, 2.8034+0.0069], alpha=0.4, color='indianred')

ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_xticklabels(['NOT Kepler', 'Gaulme Kepler', 'NOT TESS', 'Gaulme TESS', 'Gaulme et al. 2016'], rotation=20)
ax1.set_xlim([0.8, 5.2])

ax1.set_ylabel(r'Giant $\log g$ [dex]', fontsize=22)

plt.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0)
plt.savefig(fname='../figures/report/mass_radius/logg_giant.png', dpi=400)

# # # rho_avg # # #
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])

# rho_rodrigues = np.array([0.003744297, 0.000043945]) #solar ref 134.9
rho_rodrigues = np.array([0.0036785607558513134, 0.00022954244665584795])  # solar ref 136.1
rho_kjeldsen = np.array([0.0035521212, 0.000035485])
rho_gaulme = np.array([0.00285, 0.00003])

rho_not_kepler = np.array([0.00293589, 0.0, 0.00006134, 0.0])
rho_gau_kepler = np.array([0.00291372, 0.0, 0.00009851, 0.0])
rho_not_tess   = np.array([0.00305849, 0.0, 0.00010652, 0.0])
rho_gau_tess   = np.array([0.00307411, 0.0, 0.00016133, 0.0])
rho_gau_16     = np.array([0.00293, 0.0, 0.00003, 0.0])


ax1.errorbar(1, rho_not_kepler[0], yerr=rho_not_kepler[2], fmt='rp', markersize=markersize)
ax1.errorbar(2, rho_gau_kepler[0], yerr=rho_gau_kepler[2], fmt='p', color='orange', markersize=markersize)
ax1.errorbar(3, rho_not_tess[0], yerr=rho_not_tess[2], fmt='bp', markersize=markersize)
ax1.errorbar(4, rho_gau_tess[0], yerr=rho_gau_tess[2], fmt='gp', markersize=markersize)
ax1.errorbar(5, rho_gau_16[0], yerr=rho_gau_16[2], fmt='p', color='indigo', markersize=markersize)

ax1.plot(
    [0.80, 5.20],
    [rho_rodrigues[0], rho_rodrigues[0]],
    '.-', color='red', linewidth=1.0
)
ax1.plot(
    [0.80, 5.20],
    [rho_gaulme[0], rho_gaulme[0]],
    '.-', color='black', linewidth=1.0
)
ax1.plot(
    [0.80, 5.20],
    [rho_kjeldsen[0], rho_kjeldsen[0]],
    '.-', color='darkgreen', linewidth=1.0)
ax1.fill_between(
    [0.80, 5.20],
    [rho_gaulme[0]-rho_gaulme[1], rho_gaulme[0]-rho_gaulme[1]],
    [rho_gaulme[0]+rho_gaulme[1], rho_gaulme[0]+rho_gaulme[1]],
    alpha=0.4, color='grey'
)
ax1.fill_between(
    [0.80, 5.20],
    [rho_rodrigues[0]-rho_rodrigues[1], rho_rodrigues[0]-rho_rodrigues[1]],
    [rho_rodrigues[0]+rho_rodrigues[1], rho_rodrigues[0]+rho_rodrigues[1]],
    alpha=0.4, color='indianred'
)
ax1.fill_between(
    [0.80, 5.20],
    [rho_kjeldsen[0]-rho_kjeldsen[1], rho_kjeldsen[0]-rho_kjeldsen[1]],
    [rho_kjeldsen[0]+rho_kjeldsen[1], rho_kjeldsen[0]+rho_kjeldsen[1]],
    alpha=0.4, color='limegreen'
)

ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_xticklabels(['NOT Kepler', 'Gaulme Kepler', 'NOT TESS', 'Gaulme TESS', 'Gaulme et al. 2016'], rotation=20)
ax1.set_xlim([0.8, 5.2])
ax1.set_ylabel(r'Giant $\rho_{avg}$ [$\rho_{\bigodot}$]', fontsize=22)

plt.tight_layout()
plt.savefig(fname='../figures/report/mass_radius/rho.png', dpi=400)

plt.show()