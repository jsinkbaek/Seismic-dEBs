import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 22})

# # # asteroseismic estimates # # #
m_rodrigues = np.array([1.517793, 0.06921])     # dnu_ref 134.9
m_rodrigues_2 = np.array([1.4484853630502232, 0.06813140075559415])
m_rodrigues_numax = np.array([1.315274246112567, 0.07120905808555163])
m_buldgen = np.array([1.23, 0.06])
m_kjeldsen = np.array([1.68646277, 0.0740400])
m_gaulm = np.array([1.52, 0.06])
m_viani = np.array([1.451, 0.087])

r_rodrigues = np.array([8.086049, 0.1455146])   # dnu_ref 134.9
r_rodrigues_2 = np.array([7.96419, 0.146602])   # dnu_ref 134.98, numax=3141
r_rodrigues_numax = np.array([7.709110493990465, 0.1577911873091929])
r_buldgen = np.array([7.45, 0.11])
r_kjeldsen = np.array([8.5235094, 0.1441203])
r_gaulm = np.array([8.1, 0.1])
r_viani = np.array([7.97, 0.18])

m_kepler = np.array([1.2544743268, 0.01453])
r_kepler = np.array([7.532253316, 0.04258])

logg_rodrigues = np.array([2.803411, 0.00543766])   # dnu_ref 134.9
logg_kjeldsen = np.array([2.80341117, 0.00543766])
logg_gaulm = np.array([2.802, 0.004])
logg_kepler = np.array([2.7828, 0.00456])

rho_kjeldsen = np.array([0.002723463, 3.09484389E-05])
rho_gaulm = np.array([0.00285, 0.00003])
rho_rodrigues = np.array([0.002870804, 3.710783E-05])   # dnu_ref 134.9
rho_kepler = np.array([0.00293589, 0.0000435819])

# # # Mass # # #
m_rodrigues[0] = (m_rodrigues[0] - m_kepler[0])/m_kepler[0]
m_rodrigues_2[0] = (m_rodrigues_2[0] - m_kepler[0])/m_kepler[0]
m_kjeldsen[0] = (m_kjeldsen[0] - m_kepler[0])/m_kepler[0]
m_buldgen[0] = (m_buldgen[0] - m_kepler[0])/m_kepler[0]
m_rodrigues_numax[0] = (m_rodrigues_numax[0] - m_kepler[0])/m_kepler[0]
m_gaulm[0] = (m_gaulm[0] - m_kepler[0])/m_kepler[0]
m_viani[0] = (m_viani[0] - m_kepler[0])/m_kepler[0]

m_rodrigues[1] = m_rodrigues[1]/m_kepler[0]
m_rodrigues_2[1] = m_rodrigues_2[1]/m_kepler[0]
m_kjeldsen[1] = m_kjeldsen[1]/m_kepler[0]
m_buldgen[1] = m_buldgen[1]/m_kepler[0]
m_rodrigues_numax[1] = m_rodrigues_numax[1]/m_kepler[0]
m_gaulm[1] = m_gaulm[1]/m_kepler[0]
m_viani[1] = m_viani[1]/m_kepler[0]

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])

markersize = 23
elinewidth = 5
ax1.plot([0.7, 6.20], [0.0, 0.0], '--', color='grey', linewidth=elinewidth)
ax1.errorbar(1, m_rodrigues[0], yerr=m_rodrigues[1], markersize=markersize, elinewidth=elinewidth, fmt='rp')
# ax1.errorbar(2, m_rodrigues_2[0], yerr=m_rodrigues_2[1], markersize=markersize, elinewidth=elinewidth, fmt='P', color='indianred')
ax1.errorbar(4, m_rodrigues_numax[0], yerr=m_rodrigues_numax[1], markersize=markersize, elinewidth=elinewidth, fmt='s', color='orange')
ax1.errorbar(2, m_kjeldsen[0], yerr=m_kjeldsen[1], markersize=markersize, elinewidth=elinewidth, fmt='H', color='royalblue')
ax1.errorbar(3, m_gaulm[0], yerr=m_gaulm[1], markersize=markersize, elinewidth=elinewidth, fmt='gv')
ax1.errorbar(5, m_buldgen[0], yerr=m_buldgen[1], markersize=markersize, elinewidth=elinewidth, fmt='D', color='indigo')
ax1.errorbar(6, m_viani[0], yerr=m_viani[1], markersize=markersize, elinewidth=elinewidth, fmt='d', color='indianred')

ylims = ax1.get_ylim()
ax1.fill_between(
    [0.7, 6.20],
    [-m_kepler[1]/m_kepler[0], -m_kepler[1]/m_kepler[0]],
    [m_kepler[1]/m_kepler[0], m_kepler[1]/m_kepler[0]],
    alpha=0.6, color='grey'
)
ax1.set_ylim(ylims)
ax1.set_xticks([1, 2, 3, 4, 5, 6])
ax1.set_ylabel(r'$(M_{ast} - M_{dyn})/M_{dyn}$', fontsize=30)
ax1.set_xticklabels(['Rodrigues et al. (2017)',
                     # r'Rodrigues (2017), $\nu_{max, \bigodot}=3141$',
                     # r'Rodrigues (2017) $f_{\nu_{max}}=1.0489$',
                     'Kjeldsen & Bedding 1995',
                     'Gaulme et al. 2016',
                     r'Rodrigues (2017) $f_{\nu_{max}}=1.0489$',
                     'Buldgen et al. 2019',
                     r'Viani et al. 2017, $f_{\Delta\nu}=0.974$'
                     ], rotation=15)
ax1.set_xlim([0.7, 6.2])
plt.tight_layout()
plt.savefig(fname='../figures/report/mass_radius/mass_ast_pres_4.png', dpi=400)

# # # Radius # # #
r_rodrigues[0] = (r_rodrigues[0] - r_kepler[0])/r_kepler[0]
r_rodrigues_2[0] = (r_rodrigues_2[0] - r_kepler[0])/r_kepler[0]
r_kjeldsen[0] = (r_kjeldsen[0] - r_kepler[0])/r_kepler[0]
r_buldgen[0] = (r_buldgen[0] - r_kepler[0])/r_kepler[0]
r_rodrigues_numax[0] = (r_rodrigues_numax[0] - r_kepler[0])/r_kepler[0]
r_gaulm[0] = (r_gaulm[0] - r_kepler[0])/r_kepler[0]
r_viani[0] = (r_viani[0] - r_kepler[0])/r_kepler[0]

r_rodrigues[1] = r_rodrigues[1]/r_kepler[0]
r_rodrigues_2[1] = r_rodrigues_2[1]/r_kepler[0]
r_kjeldsen[1] = r_kjeldsen[1]/r_kepler[0]
r_buldgen[1] = r_buldgen[1]/r_kepler[0]
r_rodrigues_numax[1] = r_rodrigues_numax[1]/r_kepler[0]
r_gaulm[1] = r_gaulm[1]/r_kepler[0]
r_viani[1] = r_viani[1]/r_kepler[0]

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])

ax1.plot([0.7, 6.20], [0.0, 0.0], '--', color='grey', linewidth=elinewidth)
ax1.errorbar(1, r_rodrigues[0], yerr=r_rodrigues[1], markersize=markersize, elinewidth=elinewidth, fmt='rp')
# ax1.errorbar(2, r_rodrigues_2[0], yerr=r_rodrigues_2[1], markersize=markersize, elinewidth=elinewidth, fmt='P', color='indianred')
ax1.errorbar(4, r_rodrigues_numax[0], yerr=r_rodrigues_numax[1], markersize=markersize, elinewidth=elinewidth, fmt='s', color='orange')
ax1.errorbar(2, r_kjeldsen[0], yerr=r_kjeldsen[1], markersize=markersize, elinewidth=elinewidth, fmt='H', color='royalblue')
ax1.errorbar(3, r_gaulm[0], yerr=r_gaulm[1], markersize=markersize, elinewidth=elinewidth, fmt='gv')
ax1.errorbar(5, r_buldgen[0], yerr=r_buldgen[1], markersize=markersize, elinewidth=elinewidth, fmt='D', color='indigo')
ax1.errorbar(6, r_viani[0], yerr=r_viani[1], markersize=markersize, elinewidth=elinewidth, fmt='d', color='indianred')

ylims = ax1.get_ylim()
ax1.fill_between(
    [0.7, 6.20],
    [-r_kepler[1]/r_kepler[0], -r_kepler[1]/r_kepler[0]],
    [r_kepler[1]/r_kepler[0], r_kepler[1]/r_kepler[0]],
    alpha=0.6, color='grey'
)
ax1.set_ylim(ylims)
ax1.set_xticks([1, 2, 3, 4, 5, 6])
ax1.set_ylabel(r'$(R_{ast} - R_{dyn})/R_{dyn}$', fontsize=30)
ax1.set_xticklabels(['Rodrigues et al. (2017)',
                     # r'Rodrigues (2017), $\nu_{max, \bigodot}=3141$',
                     # r'Rodrigues (2017) $f_{\nu_{max}}=1.0489$',
                     'Kjeldsen & Bedding 1995',
                     'Gaulme et al. 2016',
                     r'Rodrigues (2017) $f_{\nu_{max}}=1.0489$',
                     'Buldgen et al. 2019',
                     r'Viani et al. 2017, $f_{\Delta\nu}=0.974$'
                     ], rotation=15)
ax1.set_xlim([0.7, 6.2])
plt.tight_layout()
plt.savefig(fname='../figures/report/mass_radius/radius_ast_pres_4.png', dpi=400)

# rho
rho_rodrigues[0] = (rho_rodrigues[0] - rho_kepler[0])/rho_kepler[0]
rho_kjeldsen[0] = (rho_kjeldsen[0] - rho_kepler[0])/rho_kepler[0]
rho_gaulm[0] = (rho_gaulm[0] - rho_kepler[0])/rho_kepler[0]

rho_rodrigues[1] = rho_rodrigues[1]/rho_kepler[0]
rho_kjeldsen[1] = rho_kjeldsen[1]/rho_kepler[0]
rho_gaulm[1] = rho_gaulm[1]/rho_kepler[0]

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])


ax1.errorbar(1, rho_rodrigues[0], yerr=rho_rodrigues[1], markersize=markersize, elinewidth=elinewidth, fmt='rp')
# ax1.errorbar(2, rho_rodrigues_2[0], yerr=rho_rodrigues_2[1], markersize=markersize, elinewidth=elinewidth, fmt='P', color='indianred')
ax1.errorbar(2, rho_kjeldsen[0], yerr=rho_kjeldsen[1], markersize=markersize, elinewidth=elinewidth, fmt='H', color='royalblue')
ax1.errorbar(3, rho_gaulm[0], yerr=rho_gaulm[1], markersize=markersize, elinewidth=elinewidth, fmt='gv')


ax1.fill_between(
    [0.7, 6.20],
    [-rho_kepler[1]/rho_kepler[0], -rho_kepler[1]/rho_kepler[0]],
    [rho_kepler[1]/rho_kepler[0], rho_kepler[1]/rho_kepler[0]],
    alpha=0.6, color='grey'
)
ax1.plot([0.7, 6.20], [0.0, 0.0], '--', color='grey', linewidth=elinewidth)
ylims = ax1.get_ylim()
ax1.set_ylim(ylims)
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_ylabel(r'$(\rho _{ast} - \rho _{dyn})/\rho _{dyn}$', fontsize=30)
ax1.set_xticklabels(['Rodrigues et al. (2017)',
                     # r'Rodrigues (2017), $\nu_{max, \bigodot}=3141$',
                     # r'Rodrigues (2017) $f_{\nu_{max}}=1.0489$',
                     'Kjeldsen & Bedding 1995',
                     'Gaulme et al. 2016',
                     r'Rodrigues (2017) $f_{\nu_{max}}=1.0489$',
                     'Buldgen et al. 2019',
                     ], rotation=15)
ax1.set_xlim([0.7, 3.2])
plt.tight_layout()
plt.savefig(fname='../figures/report/mass_radius/rho_ast_pres_1.png', dpi=400)

# logg
logg_rodrigues[0] = (logg_rodrigues[0] - logg_kepler[0])/logg_kepler[0]
logg_kjeldsen[0] = (logg_kjeldsen[0] - logg_kepler[0])/logg_kepler[0]
logg_gaulm[0] = (logg_gaulm[0] - logg_kepler[0])/logg_kepler[0]

logg_rodrigues[1] = logg_rodrigues[1]/logg_kepler[0]
logg_kjeldsen[1] = logg_kjeldsen[1]/logg_kepler[0]
logg_gaulm[1] = logg_gaulm[1]/logg_kepler[0]

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])


ax1.errorbar(1, logg_rodrigues[0], yerr=logg_rodrigues[1], markersize=markersize, elinewidth=elinewidth, fmt='rp')
# ax1.errorbar(2, logg_rodrigues_2[0], yerr=logg_rodrigues_2[1], markersize=markersize, elinewidth=elinewidth, fmt='P', color='indianred')
ax1.errorbar(2, logg_kjeldsen[0], yerr=logg_kjeldsen[1], markersize=markersize, elinewidth=elinewidth, fmt='H', color='royalblue')
ax1.errorbar(3, logg_gaulm[0], yerr=logg_gaulm[1], markersize=markersize, elinewidth=elinewidth, fmt='gv')


ax1.fill_between(
    [0.7, 6.20],
    [-logg_kepler[1]/logg_kepler[0], -logg_kepler[1]/logg_kepler[0]],
    [logg_kepler[1]/logg_kepler[0], logg_kepler[1]/logg_kepler[0]],
    alpha=0.6, color='grey'
)
ax1.plot([0.7, 6.20], [0.0, 0.0], '--', color='grey', linewidth=elinewidth)
ylims = ax1.get_ylim()
ax1.set_ylim(ylims)
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_ylabel(r'$(\log g_{ast} - \log g_{dyn})/\log g_{dyn}$', fontsize=30)
ax1.set_xticklabels(['Rodrigues et al. (2017)',
                     # r'Rodrigues (2017), $\nu_{max, \bigodot}=3141$',
                     # r'Rodrigues (2017) $f_{\nu_{max}}=1.0489$',
                     'Kjeldsen & Bedding 1995',
                     'Gaulme et al. 2016',
                     'Buldgen et al. 2019',
                     r'Rodrigues (2017) $f_{\nu_{max}}=1.0489$',
                     ], rotation=15)
ax1.set_xlim([0.7, 3.2])
plt.tight_layout()
plt.savefig(fname='../figures/report/mass_radius/logg_ast_pres_1.png', dpi=400)

plt.show()
