import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from Binary_Analysis.block_bootstrap import block_bootstrap as boot

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/block_bootstrap/')

period = 63.3270949830
nblocks = 4      # per eclipse

lc_model = np.loadtxt('work/model.KEPLER.NOT')
lc_err = lc_model[:, 2]
time = lc_model[:, 0]
model = lc_model[:, 4]
residual = lc_model[:, 5]
phase = np.mod(time, period)/period
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
        sub_arrays_secondary.append(
            np.array([time[start:end], model[start:end]]).T
        )
        diff_sec.append(time[end-1]-time[start])
        len_sec.append(time[start:end].size)
        # ax1.plot(phase[start:end], model[start:end]+0.002*i, '*')

    elif indices[i] in indices_primary:
        sub_arrays_primary.append(
            np.array([time[start:end], model[start:end]]).T
        )
        diff_prim.append(time[end-1]-time[start])
        len_prim.append(time[start:end].size)
        # ax2.plot(phase[start:end], model[start:end]+0.002*i, '*')

    else:
        raise ValueError('index not in either')

# plt.show()

primary_time = np.median(diff_prim)
secondary_time = np.median(diff_sec)

nblocks_sec = nblocks  # int(np.rint(secondary_time/block_length))
nblocks_prim = nblocks  # int(np.rint(primary_time/block_length))


sub_arrays_split = []

# Concatenate all eclipses to one list of arrays
blen_prim = []
blen_sec = []
for i in range(0, len(sub_arrays_primary)):
    blen_prim.append(sub_arrays_primary[i].shape[0]/nblocks)
    sub_arrays_split += np.array_split(sub_arrays_primary[i], nblocks_prim, axis=0)
for i in range(0, len(sub_arrays_secondary)):
    blen_sec.append(sub_arrays_secondary[i].shape[0]/nblocks)
    sub_arrays_split += np.array_split(sub_arrays_secondary[i], nblocks_sec, axis=0)


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
               'radius_B', 'logg_A', 'logg_B', 'sma_rsun', 'period', 'lum_ratio']


mean_vals, std_vals, vals = boot.residual_block_bootstrap(
    lc_blocks, residual, lc_err, rvA, rvB, 4000, param_names, 10, rvA_model, rvB_model,
    infile_name='infile.residual.NOT.KEPLER', draw_random_rv_obs=False
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
np.savetxt('vals.NOT.KEPLER.resi.4', vals)
