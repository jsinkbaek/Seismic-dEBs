import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import math

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/')


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, math.sqrt(variance)


matplotlib.rcParams.update({'font.size': 25})
os.chdir('Binary_Analysis/JKTEBOP/NOT/kepler_LTF/')

model_filename = 'model.out'
rva = 'rvA.out'
rvb = 'rvB.out'
phase_model, rv_Am, rv_Bm = np.loadtxt(model_filename, usecols=(0, 6, 7), unpack=True)
rva = np.loadtxt(rva)
rvb = np.loadtxt(rvb)

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(5, 1)
ax23 = fig.add_subplot(gs[3:5, 0])
ax1 = fig.add_subplot(gs[0:3, 0])
ax2 = fig.add_subplot(gs[3, 0])
ax3 = fig.add_subplot(gs[4, 0])
ax3.set_xlabel('Orbital Phase')
ax1.set_ylabel('Radial Velocity (km/s)')
ax23.set_ylabel('O-C')
# ax2.set_ylabel('O - C')
# ax3.set_ylabel('O - C')
ax1.set_xlim([0, 1.0])
ax2.set_xlim([0, 1.0])
ax3.set_xlim([0, 1.0])

a_mask = rva[:, 0] > 59000
b_mask = rvb[:, 0] > 59000

ax1.errorbar(rva[:, 3], rva[:, 1], yerr=rva[:, 2], fmt='*', color='blue', markersize=6)
ax1.errorbar(rvb[:, 3], rvb[:, 1], yerr=rvb[:, 2], fmt='*', color='red', markersize=6)
ax1.plot(phase_model, rv_Am, '--', alpha=0.8, color='grey')
ax1.plot(phase_model, rv_Bm, '--', alpha=0.8, color='grey')
ax1.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
ax1.tick_params(
    axis='y',
    which='major',
    left=True,
    labelleft=True,
    labelsize=22,
)
ax2.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
ax2.tick_params(
    axis='y',
    which='major',
    left=True,
    labelleft=True,
    labelsize=18,
)

ax3.tick_params(
    axis='x',
    which='minor',
    bottom=True,
    top=False,
    labelbottom=True
)
ax3.tick_params(
    axis='y',
    which='major',
    left=True,
    labelleft=True,
    labelsize=18,
)
ax23.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelcolor='w'
)
ax2.yaxis.set_ticks([-0.1, 0.1])
ax3.yaxis.set_ticks([-1, 1])
ax3.xaxis.set_minor_locator(AutoMinorLocator())


ax2.errorbar(rva[:, 3], rva[:, 5], yerr=rva[:, 2], fmt='*', color='blue')
ax3.errorbar(rvb[:, 3], rvb[:, 5], yerr=rvb[:, 2], fmt='*', color='red')
ax2.plot([0, 1], [0, 0], '--', color='black', alpha=0.8)
ax3.plot([0, 1], [0, 0], '--', color='black', alpha=0.8)
# _, wstd_a = weighted_avg_and_std(rva[:, 5], 1/rva[:, 2]**2)
# _, wstd_b = weighted_avg_and_std(rvb[:, 5], 1/rvb[:, 2]**2)
wstd_a = np.std(rva[:, 5])
wstd_b = np.std(rvb[:, 5])
ax2.fill_between([0.0, 1.0], [wstd_a, wstd_a], [-wstd_a, -wstd_a],
                 color='grey', alpha=0.4)
ax3.fill_between([0.0, 1.0], [wstd_b, wstd_b], [-wstd_b, -wstd_b],
                 color='grey', alpha=0.4)
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

