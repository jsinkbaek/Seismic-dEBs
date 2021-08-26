import numpy as np
import matplotlib.pyplot as plt
import os
from Binary_Analysis.block_bootstrap import block_bootstrap as boot

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/block_bootstrap/')

period = 63.3271047458

lc = np.loadtxt('work/lc.KEPLER')
phase = np.mod(lc[:, 0], period)/period
phase_primary_eclipse = np.mod((54998.2347431865 - 54976.6348) + 54976.6348, period)/period
phase_secondary_eclipse = (phase_primary_eclipse + 0.6589217480)-1

indices = np.argwhere((np.diff(phase) > 0.2) | (np.diff(phase) < -0.03))[:, 0] + 1
indices_secondary = np.argwhere((np.diff(phase) > 0.2) | ((np.diff(phase) < -0.03) & (phase[:-1] < 0.3)))[:, 0] + 1
indices_primary = np.argwhere(((np.diff(phase) < -0.03) & (phase[:-1] > 0.3)))[:, 0] + 1
sub_arrays_secondary = []
sub_arrays_primary = []
midtimes_secondary = []
midtimes_primary = []
for i in range(0, len(indices)):
    if i == 0:
        start = 0; end = indices[i]
    else:
        start = indices[i-1]; end = indices[i]
    if indices[i] in indices_secondary:
        sub_arrays_secondary.append(lc[start:end, :])
        phase_temp = np.mod(lc[start:end, 0], period)/period
        relative_phase_diff = phase_temp - phase_secondary_eclipse
        midtime = np.median(lc[start:end, 0] - relative_phase_diff*period)
        midtimes_secondary.append(midtime)
    elif indices[i] in indices_primary:
        sub_arrays_primary.append(lc[start:end, :])
        phase_temp = np.mod(lc[start:end, 0], period) / period
        relative_phase_diff = phase_temp - phase_primary_eclipse
        midtime = np.median(lc[start:end, 0] - relative_phase_diff * period)
        midtimes_primary.append(midtime)
    else:
        raise ValueError('index not in either')

midtimes_secondary = np.asarray(midtimes_secondary)
midtimes_primary = np.asarray(midtimes_primary)

row_size_1 = np.max([x.shape[0] for x in sub_arrays_secondary])
row_size_2 = np.max(np.max([x.shape[0] for x in sub_arrays_primary]))
row_size = np.max([row_size_1, row_size_2])
lc_blocks = np.empty((row_size, 3, len(sub_arrays_secondary)+len(sub_arrays_primary)))
lc_blocks[:] = np.nan
for i in range(0, len(sub_arrays_secondary)):
    current_array = sub_arrays_secondary[i]
    lc_blocks[0:current_array.shape[0], :, i] = current_array
for i in range(len(sub_arrays_secondary), len(sub_arrays_primary)+len(sub_arrays_secondary)):
    current_array = sub_arrays_primary[i-len(sub_arrays_secondary)]
    lc_blocks[0:current_array.shape[0], :, i] = current_array


rvA = np.loadtxt('work/rvA.dat')
rvB = np.loadtxt('work/rvB.dat')

rvA_model = np.loadtxt('work/rvA.model', unpack=True, usecols=4)
rvB_model = np.loadtxt('work/rvB.model', unpack=True, usecols=4)

print(rvA_model)

midtimes = [midtimes_secondary, midtimes_primary]


param_names = ['sb_ratio', 'sum_radii', 'ratio_radii', 'incl', 'ecc', 'perilong', 'light_scale_factor',
               'ephemeris_tbase', 'rv_amp_A', 'rv_amp_B', 'system_rv_A', 'system_rv_B', 'mass_A', 'mass_B', 'radius_A',
               'radius_B', 'logg_A', 'logg_B']

mean_vals, std_vals, vals = boot.block_bootstrap(
    lc_blocks, rvA, rvB, 50, param_names, n_jobs=12, block_midtime=midtimes, rvA_model=rvA_model, rvB_model=rvB_model
)


print('{:>14}'.format('Mean Value'), '\t', '{:>14}'.format('STD'), '\t', '{:<20}'.format('Parameter'))
for i in range(0, len(param_names)):
    print(f'{mean_vals[i]:14.5f}', '\t', f'{std_vals[i]:14.7f}', '\t',f'{param_names[i]: <20}')
np.savetxt('vals', vals)
