import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 18})

# # # asteroseismic estimates # # #
m_rodrigues = np.array([1.517793, 0.06921])     # dnu_ref 134.9
m_rodrigues_2 = np.array([1.4484853630502232, 0.06813140075559415])
m_rodrigues_numax = np.array([1.315274246112567, 0.07120905808555163])
m_buldgen = np.array([1.23, 0.06])
m_kjeldsen = np.array([1.68646277, 0.0740400])
m_gaulm = np.array([1.52, 0.06])

r_rodrigues = np.array([8.086049, 0.1455146])   # dnu_ref 134.9
r_rodrigues_2 = np.array([7.96419, 0.146602])   # dnu_ref 134.98, numax=3141
r_rodrigues_numax = np.array([7.709110493990465, 0.1577911873091929])
r_buldgen = np.array([7.45, 0.11])
r_kjeldsen = np.array([8.5235094, 0.1441203])
r_gaulm = np.array([8.1, 0.1])

m_kepler = np.array([1.2544743268, 0.01453])
r_kepler = np.array([7.532253316, 0.04258])

# # # Mass # # #
m_rodrigues[0] = (m_rodrigues[0] - m_kepler[0])/m_kepler[0]
m_rodrigues_2[0] = (m_rodrigues_2[0] - m_kepler[0])/m_kepler[0]
m_kjeldsen[0] = (m_kjeldsen[0] - m_kepler[0])/m_kepler[0]
m_buldgen[0] = (m_buldgen[0] - m_kepler[0])/m_kepler[0]
m_rodrigues_numax[0] = (m_rodrigues_numax[0] - m_kepler[0])/m_kepler[0]
m_gaulm[0] = (m_gaulm[0] - m_kepler[0])/m_kepler[0]

m_rodrigues[1] = m_rodrigues[1]/m_kepler[0]
m_rodrigues_2[1] = m_rodrigues_2[1]/m_kepler[0]
m_kjeldsen[1] = m_kjeldsen[1]/m_kepler[0]
m_buldgen[1] = m_buldgen[1]/m_kepler[0]
m_rodrigues_numax[1] = m_rodrigues_numax[1]/m_kepler[0]
m_gaulm[1] = m_gaulm[1]/m_kepler[0]

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])

markersize = 15
ax1.errorbar(1, m_rodrigues[0], yerr=m_rodrigues[1], markersize=markersize, fmt='rp')
ax1.errorbar(2, m_rodrigues_2[0], yerr=m_rodrigues_2[1], markersize=markersize, fmt='P', color='indianred')
ax1.errorbar(3, m_rodrigues_numax[0], yerr=m_rodrigues_numax[1], markersize=markersize, fmt='s', color='orange')
ax1.errorbar(4, m_kjeldsen[0], yerr=m_kjeldsen[1], markersize=markersize, fmt='H', color='royalblue')
ax1.errorbar(5, m_gaulm[0], yerr=m_gaulm[1], markersize=markersize, fmt='gv')
ax1.errorbar(6, m_buldgen[0], yerr=m_buldgen[1], markersize=markersize, fmt='D', color='indigo')

ylims = ax1.get_ylim()
ax1.fill_between(
    [0.7, 6.20],
    [-m_kepler[1]/m_kepler[0], -m_kepler[1]/m_kepler[0]],
    [m_kepler[1]/m_kepler[0], m_kepler[1]/m_kepler[0]],
    alpha=0.4, color='grey'
)
ax1.set_ylim(ylims)
ax1.set_xticks([1, 2, 3, 4, 5, 6])
ax1.set_ylabel(r'$(M_{ast} - M_{dyn})/M_{dyn}$', fontsize=22)
ax1.set_xticklabels(['Rodrigues et al. (2017)',
                     r'Rodrigues (2017), $\nu_{max, \bigodot}=3141$',
                     r'Rodrigues (2017) $f_{\nu_{max}}=1.0489$',
                     'Kjeldsen & Bedding 1995', 'Gaulme et al. 2016', 'Buldgen et al. 2019'], rotation=15)
ax1.set_xlim([0.7, 6.2])
plt.tight_layout()
plt.savefig(fname='../figures/report/mass_radius/mass_ast.png', dpi=400)

# # # Radius # # #
r_rodrigues[0] = (r_rodrigues[0] - r_kepler[0])/r_kepler[0]
r_rodrigues_2[0] = (r_rodrigues_2[0] - r_kepler[0])/r_kepler[0]
r_kjeldsen[0] = (r_kjeldsen[0] - r_kepler[0])/r_kepler[0]
r_buldgen[0] = (r_buldgen[0] - r_kepler[0])/r_kepler[0]
r_rodrigues_numax[0] = (r_rodrigues_numax[0] - r_kepler[0])/r_kepler[0]
r_gaulm[0] = (r_gaulm[0] - r_kepler[0])/r_kepler[0]

r_rodrigues[1] = r_rodrigues[1]/r_kepler[0]
r_rodrigues_2[1] = r_rodrigues_2[1]/r_kepler[0]
r_kjeldsen[1] = r_kjeldsen[1]/r_kepler[0]
r_buldgen[1] = r_buldgen[1]/r_kepler[0]
r_rodrigues_numax[1] = r_rodrigues_numax[1]/r_kepler[0]
r_gaulm[1] = r_gaulm[1]/r_kepler[0]

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])


ax1.errorbar(1, r_rodrigues[0], yerr=r_rodrigues[1], markersize=markersize, fmt='rp')
ax1.errorbar(2, r_rodrigues_2[0], yerr=r_rodrigues_2[1], markersize=markersize, fmt='P', color='indianred')
ax1.errorbar(3, r_rodrigues_numax[0], yerr=r_rodrigues_numax[1], markersize=markersize, fmt='s', color='orange')
ax1.errorbar(4, r_kjeldsen[0], yerr=r_kjeldsen[1], markersize=markersize, fmt='H', color='royalblue')
ax1.errorbar(5, r_gaulm[0], yerr=r_gaulm[1], markersize=markersize, fmt='gv')
ax1.errorbar(6, r_buldgen[0], yerr=r_buldgen[1], markersize=markersize, fmt='D', color='indigo')

ylims = ax1.get_ylim()
ax1.fill_between(
    [0.7, 6.20],
    [-r_kepler[1]/r_kepler[0], -r_kepler[1]/r_kepler[0]],
    [r_kepler[1]/r_kepler[0], r_kepler[1]/r_kepler[0]],
    alpha=0.4, color='grey'
)
ax1.set_ylim(ylims)
ax1.set_xticks([1, 2, 3, 4, 5, 6])
ax1.set_ylabel(r'$(R_{ast} - R_{dyn})/R_{dyn}$', fontsize=22)
ax1.set_xticklabels(['Rodrigues et al. (2017)',
                     r'Rodrigues (2017), $\nu_{max, \bigodot}=3141$',
                     r'Rodrigues (2017) $f_{\nu_{max}}=1.0489$',
                     'Kjeldsen & Bedding 1995', 'Gaulme et al. 2016', 'Buldgen et al. 2019'], rotation=15)
ax1.set_xlim([0.7, 6.2])
plt.tight_layout()
plt.savefig(fname='../figures/report/mass_radius/radius_ast.png', dpi=400)
plt.show()
