import numpy as np
import matplotlib.pyplot as plt
import os
from Binary_Analysis.block_bootstrap import block_bootstrap as boot

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/block_bootstrap/')

period = 63.327

lc = np.loadtxt('work/lc.KEPLER')
phase = np.mod(lc[:, 0], period)/period

# plt.plot(phase, lc[:, 1], '*')
# plt.show()
# plt.plot(phase[:-1], np.diff(phase), '*')
# plt.show()

indices = np.argwhere((np.diff(phase) > 0.2) | (np.diff(phase) < -0.03))[:, 0] + 1
sub_arrays = np.split(lc, indices, axis=0)
row_size = np.max([x.shape[0] for x in sub_arrays])
lc_blocks = np.empty((row_size, 3, len(sub_arrays)))
lc_blocks[:] = np.nan
for i in range(0, len(sub_arrays)):
    current_array = sub_arrays[i]
    lc_blocks[0:current_array.shape[0], :, i] = current_array

rvA = np.loadtxt('work/rvA.dat')
rvB = np.loadtxt('work/rvB.dat')

param_names = ['sb_ratio', 'sum_radii', 'ratio_radii', 'incl', 'ecc', 'perilong', 'light_scale_factor', 'period',
               'ephemeris_tbase', 'rv_amp_A', 'rv_amp_B', 'system_rv_A', 'system_rv_B']

mean_vals, std_vals = boot.block_bootstrap(lc_blocks, rvA, rvB, 50, param_names, n_jobs=8)
print(mean_vals)
print(std_vals)


