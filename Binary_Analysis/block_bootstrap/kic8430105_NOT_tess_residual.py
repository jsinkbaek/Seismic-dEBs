import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from Binary_Analysis.block_bootstrap import block_bootstrap as boot
from Binary_Analysis.block_bootstrap.kic8430105_convenience_functions import split_tess_residual_lightcurve

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/block_bootstrap/')
period = 63.3270949830
lc_model = np.loadtxt('work/model.TESS.NOT')
lc_err = lc_model[:, 2]
residual = lc_model[:, 5]
rvA = np.loadtxt('work/rvA.NOT.dat')
rvB = np.loadtxt('work/rvB.NOT.dat')
rvA_model = np.loadtxt('work/rvA.NOT.TESS.model', unpack=True, usecols=4)
rvB_model = np.loadtxt('work/rvB.NOT.TESS.model', unpack=True, usecols=4)
param_names = [
    'sb_ratio', 'sum_radii', 'ratio_radii', 'incl', 'ecc', 'perilong', 'light_scale_factor', 'ephemeris_tbase',
    'rv_amp_A', 'rv_amp_B', 'system_rv_A', 'system_rv_B', 'mass_A', 'mass_B', 'radius_A', 'radius_B', 'logg_A',
    'logg_B', 'sma_rsun', 'lum_ratio', 'rho_A', 'rho_B', 'chisqr'
]
n_iter = 2500
n_blocks_list = [2, 4, 6, 8, 10, 12, 14, 16]
n_cores = 10
infile_name = 'infile.residual.NOT.TESS'
draw_random_rv_obs=False

# # Run with different block sizes
print('Residual block bootstrap with NOT RVs and TESS LC, KIC 8430105.')
for n_blocks in n_blocks_list:
    mean_vals, std_vals, vals = boot.residual_block_bootstrap(
        lc_model, residual, lc_err, rvA, rvB, n_iter, param_names, n_cores, rvA_model, rvB_model, draw_random_rv_obs,
        infile_name, split_function=split_tess_residual_lightcurve, split_args=(n_blocks, )
    )
    print('')
    print('n_blocks: ', n_blocks)
    print('{:>14}'.format('Mean Value'), '\t', '{:>14}'.format('STD'), '\t', '{:<20}'.format('Parameter'))
    for i in range(0, len(param_names)):
        print(f'{mean_vals[i]:14.5f}', '\t', f'{std_vals[i]:14.7f}', '\t', f'{param_names[i]: <20}')
    np.savetxt(f'vals.NOT.TESS.resi.{n_blocks}', vals)