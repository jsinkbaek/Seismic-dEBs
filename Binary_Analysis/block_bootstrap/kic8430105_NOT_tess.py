import numpy as np
import matplotlib.pyplot as plt
import os
from Binary_Analysis.block_bootstrap import block_bootstrap as boot

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/block_bootstrap/')

period = 63.3270949830

lc = np.loadtxt('work/lc.TESS')

mask_secondary = lc[:, 0] < np.mean(lc[:, 0])
lc_block_secondary = lc[mask_secondary, :]
lc_block_primary = lc[~mask_secondary, :]

lc_blocks = np.empty((np.max([lc_block_secondary[:, 0].size, lc_block_primary[:, 0].size]), 3, 2), dtype=lc.dtype)
lc_blocks[:, :, :] = np.nan
lc_blocks[0:lc_block_secondary[:, 0].size, :, 0] = lc_block_secondary
lc_blocks[0:lc_block_primary[:, 0].size, :, 1] = lc_block_primary
# lc_blocks = lc.reshape((lc.shape[0], lc.shape[1], 1))

rvA = np.loadtxt('work/rvA.NOT.dat')
rvB = np.loadtxt('work/rvB.NOT.dat')

rvA_model = np.loadtxt('work/rvA.NOT.model', unpack=True, usecols=4)
rvB_model = np.loadtxt('work/rvB.NOT.model', unpack=True, usecols=4)


param_names = ['sb_ratio', 'sum_radii', 'ratio_radii', 'incl', 'ecc', 'perilong', 'light_scale_factor',
               'ephemeris_tbase', 'rv_amp_A', 'rv_amp_B', 'system_rv_A', 'system_rv_B', 'mass_A', 'mass_B', 'radius_A',
               'radius_B', 'logg_A', 'logg_B', 'sma_rsun']

mean_vals, std_vals, vals = boot.block_bootstrap_variable_moving_blocks(
    lc_blocks, rvA, rvB, 3000, param_names,
    subgroup_divisions=(20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                        50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                        60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                        70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80), period=period,
    n_jobs=10,
    rvA_model=rvA_model, rvB_model=rvB_model,
    infile_name='infile.default.NOT.TESS'
)


print('{:>14}'.format('Mean Value'), '\t', '{:>14}'.format('STD'), '\t', '{:<20}'.format('Parameter'))
for i in range(0, len(param_names)):
    print(f'{mean_vals[i]:14.5f}', '\t', f'{std_vals[i]:14.7f}', '\t',f'{param_names[i]: <20}')
np.savetxt('vals.NOT.TESS', vals)
