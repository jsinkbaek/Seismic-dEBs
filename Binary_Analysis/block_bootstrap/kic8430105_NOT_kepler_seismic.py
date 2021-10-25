import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from Binary_Analysis.block_bootstrap import block_bootstrap as boot

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/block_bootstrap/')

period = 63.3270949830
block_length = 0.3      # days

lc = np.loadtxt('work/lc.KEPLER')
phase = np.mod(lc[:, 0], period)/period
# plt.plot(np.diff(phase), '*')
# plt.show(block=True)

indices = np.argwhere((np.diff(phase) > 0.2) | (np.diff(phase) < -0.002))[:, 0] +1
indices_secondary = np.argwhere((np.diff(phase) > 0.2) | ((np.diff(phase) < -0.002) & (phase[:-1] < 0.3)))[:, 0] +1
indices_primary = np.argwhere(((np.diff(phase) < -0.002) & (phase[:-1] > 0.3)))[:, 0] +1
sub_arrays_secondary = []
sub_arrays_primary = []
diff_prim = []
diff_sec = []
len_prim = []
len_sec = []
# fig1 = plt.figure()
# fig2 = plt.figure()
# ax1 = fig1.add_subplot()
# ax2 = fig2.add_subplot()
for i in range(0, len(indices)):
    if i == 0:
        start = 0; end = indices[i]
    else:
        start = indices[i-1]; end = indices[i]
    if indices[i] in indices_secondary:
        sub_arrays_secondary.append(lc[start:end, :])
        diff_sec.append(lc[end-1, 0]-lc[start, 0])
        len_sec.append(lc[start:end, 0].size)
        # ax1.plot(np.mod(lc[start:end, 0], period), lc[start:end, 1]+0.002*i, '*')

    elif indices[i] in indices_primary:
        sub_arrays_primary.append(lc[start:end, :])
        diff_prim.append(lc[end-1, 0]-lc[start, 0])
        len_prim.append(lc[start:end, 0].size)
        # ax2.plot(np.mod(lc[start:end, 0], period), lc[start:end, 1]+0.002*i, '*')

    else:
        raise ValueError('index not in either')

# plt.show()

primary_time = np.median(diff_prim)
secondary_time = np.median(diff_sec)

primary_time/block_length
nblocks_sec = int(np.rint(secondary_time/block_length))
nblocks_prim = int(np.rint(primary_time/block_length))

sub_arrays_split = []

# Concatenate all eclipses to one list of arrays
for i in range(0, len(sub_arrays_primary)):
    sub_arrays_split += np.array_split(sub_arrays_primary[i], nblocks_prim, axis=0)
for i in range(0, len(sub_arrays_secondary)):
    sub_arrays_split += np.array_split(sub_arrays_secondary[i], nblocks_sec, axis=0)

row_size = np.max([x.shape[0] for x in sub_arrays_split])
lc_blocks = np.empty((row_size, 3, len(sub_arrays_split)))
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
               'radius_B', 'logg_A', 'logg_B', 'sma_rsun', 'period', 'lum_ratio']

mean_vals, std_vals, vals = boot.block_bootstrap(
    lc_blocks, rvA, rvB, 10000, param_names, n_jobs=10, block_midtime=None, rvA_model=rvA_model, rvB_model=rvB_model,
    infile_name='infile.default.NOT.KEPLER'
)

# import Binary_Analysis.block_bootstrap.jktebop_io_interface as jio

# job_results = []
# for i in range(0, 9852):
#     job_results.append(
#         jio.pull_parameters_from_outfile(
#             'work/'+str(i)+'/',
#             param_names
#         )
#     )
# mean_vals, std_vals, vals = boot.evaluate_runs(job_results)

print('{:>14}'.format('Mean Value'), '\t', '{:>14}'.format('STD'), '\t', '{:<20}'.format('Parameter'))
for i in range(0, len(param_names)):
    print(f'{mean_vals[i]:14.5f}', '\t', f'{std_vals[i]:14.7f}', '\t',f'{param_names[i]: <20}')
np.savetxt('vals.NOT.KEPLER.seism', vals)
