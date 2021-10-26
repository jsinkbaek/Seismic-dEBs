import numpy as np
import matplotlib.pyplot as plt
import os
from Binary_Analysis.block_bootstrap import block_bootstrap as boot

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/block_bootstrap/')

period = 63.3270949830
block_length = 0.3
# nblocks =

# lc = np.loadtxt('work/lc.TESS')
lc_model = np.loadtxt('work/model.TESS.NOT')
lc_err = lc_model[:, 2]
time = lc_model[:, 0]
model = lc_model[:, 4]
residual = lc_model[:, 5]
phase = np.mod(time, period)/period

time_diff = np.median(np.diff(time))*time.size
nblocks = int(np.rint(time_diff/block_length))

sub_arrays_split = np.array_split(np.array([time, model]).T, nblocks, axis=0)


row_size = np.max([x.shape[0] for x in sub_arrays_split])
lc_blocks = np.empty((row_size, 2, len(sub_arrays_split)))
lc_blocks[:] = np.nan
for i in range(0, len(sub_arrays_split)):
    current_array = sub_arrays_split[i]
    lc_blocks[0:current_array.shape[0], :, i] = current_array

rvA = np.loadtxt('work/rvA.NOT.dat')
rvB = np.loadtxt('work/rvB.NOT.dat')

rvA_model = np.loadtxt('work/rvA.NOT.model', unpack=True, usecols=4)
rvB_model = np.loadtxt('work/rvB.NOT.model', unpack=True, usecols=4)


param_names = ['sb_ratio', 'sum_radii', 'ratio_radii', 'incl', 'ecc', 'perilong', 'light_scale_factor',
               'ephemeris_tbase', 'rv_amp_A', 'rv_amp_B', 'system_rv_A', 'system_rv_B', 'mass_A', 'mass_B', 'radius_A',
               'radius_B', 'logg_A', 'logg_B', 'sma_rsun', 'lum_ratio']

mean_vals, std_vals, vals = boot.residual_block_bootstrap(
    lc_blocks, residual, lc_err, rvA, rvB, 10000, param_names, 10, rvA_model, rvB_model,
    infile_name='infile.residual.NOT.TESS', draw_random_rv_obs=True
)


print('{:>14}'.format('Mean Value'), '\t', '{:>14}'.format('STD'), '\t', '{:<20}'.format('Parameter'))
for i in range(0, len(param_names)):
    print(f'{mean_vals[i]:14.5f}', '\t', f'{std_vals[i]:14.7f}', '\t',f'{param_names[i]: <20}')
np.savetxt('vals.NOT.TESS.resi', vals)
