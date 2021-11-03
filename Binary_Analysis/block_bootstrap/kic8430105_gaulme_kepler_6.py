import numpy as np
import matplotlib.pyplot as plt
import os
from Binary_Analysis.block_bootstrap import block_bootstrap as boot

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/block_bootstrap/')

period = 63.3271348987
nblocks = 6

lc = np.loadtxt('work/lc.KEPLER')
phase = np.mod(lc[:, 0], period)/period

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
            lc[start:end, :]
        )
        diff_sec.append(lc[end-1, 0]-lc[start, 0])
        len_sec.append(lc[start:end, 0].size)
        # ax1.plot(phase[start:end], model[start:end]+0.002*i, '*')

    elif indices[i] in indices_primary:
        sub_arrays_primary.append(
            lc[start:end, :]
        )
        diff_sec.append(lc[end - 1, 0] - lc[start, 0])
        len_sec.append(lc[start:end, 0].size)
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
lc_blocks = np.empty((row_size, 3, len(sub_arrays_split)))
lc_blocks[:] = np.nan
for i in range(0, len(sub_arrays_split)):
    current_array = sub_arrays_split[i]
    lc_blocks[0:current_array.shape[0], :, i] = current_array


rvA = np.loadtxt('work/rvA.gaulme.dat')
rvB = np.loadtxt('work/rvB.gaulme.dat')

rvA_model = np.loadtxt('work/rvA.gaulme.model', unpack=True, usecols=4)
rvB_model = np.loadtxt('work/rvB.gaulme.model', unpack=True, usecols=4)


param_names = ['sb_ratio', 'sum_radii', 'ratio_radii', 'incl', 'ecc', 'perilong', 'light_scale_factor',
               'ephemeris_tbase', 'rv_amp_A', 'rv_amp_B', 'system_rv_A', 'system_rv_B', 'mass_A', 'mass_B', 'radius_A',
               'radius_B', 'logg_A', 'logg_B', 'sma_rsun', 'period', 'lum_ratio']

mean_vals, std_vals, vals = boot.block_bootstrap(
    lc_blocks, rvA, rvB, 4000, param_names, n_jobs=10, block_midtime=None, rvA_model=rvA_model, rvB_model=rvB_model,
    draw_random_rv_obs=False, infile_name='infile.default.gaulme.KEPLER'
)

print('{:>14}'.format('Mean Value'), '\t', '{:>14}'.format('STD'), '\t', '{:<20}'.format('Parameter'))
for i in range(0, len(param_names)):
    print(f'{mean_vals[i]:14.5f}', '\t', f'{std_vals[i]:14.7f}', '\t',f'{param_names[i]: <20}')
np.savetxt('vals.gaulme.KEPLER.6', vals)
