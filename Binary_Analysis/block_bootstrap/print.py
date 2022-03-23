import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 22})

param_names = ['sb_ratio', 'sum_radii', 'ratio_radii', 'incl', 'ecc', 'perilong', 'light_scale_factor',
               'ephemeris_tbase', 'rv_amp_A', 'rv_amp_B', 'system_rv_A', 'system_rv_B', 'mass_A', 'mass_B', 'radius_A',
               'radius_B', 'logg_A', 'logg_B', 'sma_rsun', 'period', 'lum_ratio', 'rho_A', 'rho_B', 'chisqr', 'ecosw',
               'esinw', 'limbd_A1']

path_bulk = 'vals.NOT.PDCSAP.resi.'
block_sizes = ['1', '2', '3', '4', '5', '6', '7', '8']
values = []
stds_all_blocks = []
means_all_blocks = []
mean_all_blocks = []
std_all_blocks = []
for i in range(0, len(block_sizes)):
    vals_temp = np.loadtxt(path_bulk+block_sizes[i])
    vals_temp = vals_temp[~np.isnan(vals_temp).any(axis=1), :]
    values.append(np.copy(vals_temp))
    stds_temp = np.asarray([np.std(vals_temp[:x, :], axis=0) for x in range(10, vals_temp[:, 0].size)])
    stds_all_blocks.append(np.copy(stds_temp))
    means_temp = np.asarray([np.mean(vals_temp[:x, :], axis=0) for x in range(10, vals_temp[:, 0].size)])
    means_all_blocks.append(np.copy(means_temp))
    mean_all_blocks.append(np.mean(vals_temp, axis=0))
    std_all_blocks.append(np.std(vals_temp, axis=0))

std_all_blocks = np.asarray(std_all_blocks)
mean_all_blocks = np.asarray(mean_all_blocks)

np.set_printoptions(precision=10, suppress=True)

print('{:<20}'.format('Parameter'), '\t', '{:>14}'.format('Max STD'), '\t', '{:>14}'.format('Min STD'), '\t',
      '{:>14}'.format('Mean STD'))
where_max = np.empty((len(param_names)), dtype=int)
for i in range(0, len(param_names)):
    where_max[i] = np.argmax(std_all_blocks[:, i])
    print(f'{param_names[i]: <20}', '\t', f'{np.max(std_all_blocks[:, i]):14.7f}', '\t',
          f'{np.min(std_all_blocks[:, i]):14.5f}', '\t', f'{np.mean(std_all_blocks[:, i]):14.5f}')

for i in range(12, 16):
    plt.figure()
    plt.plot(range(10, 3000), stds_all_blocks[where_max[i]][:, i])
    plt.xlabel('Iteration')
    plt.ylabel('Standard deviation')
plt.show()


