import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sigmaclip

os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/block_bootstrap/')
path = 'vals.gaulme.KEPLER.1'

vals = np.loadtxt(path)
vals = np.append(vals, np.loadtxt('vals.gaulme.KEPLER.2'), axis=0)
vals = np.append(vals, np.loadtxt('vals.gaulme.KEPLER.3'), axis=0)
# vals = np.append(vals, np.loadtxt('vals.NOT.KEPLER.4'), axis=0)

vals = vals[~np.isnan(vals[:, -2]), :]      # since they seem to be wack
clip_mask = np.zeros((vals[:, 0].size, ), dtype=bool)
for i in range(0, vals[0, :].size):
    clipped, lower, upper = sigmaclip(vals[:, i], low=10.0, high=10.0)
    clip_mask = clip_mask | ((vals[:, i] < lower) | (vals[:, i] > upper))
vals_old = np.copy(vals)
vals = vals[~clip_mask, :]

stds = [np.std(vals[:x, :], axis=0) for x in range(10, vals[:, 0].size)]
stds_old = [np.std(vals_old[:x, :], axis=0) for x in range(10, vals_old[:, 0].size)]

means = [np.mean(vals[:x, :], axis=0) for x in range(10, vals[:, 0].size)]

stds = np.asarray(stds)
stds_old = np.asarray(stds_old)
means = np.asarray(means)

np.set_printoptions(precision=10, suppress=True)

print('STDS')
print(np.std(vals, axis=0))
print('\nMeans')
print(np.mean(vals, axis=0))
plt.plot(stds)

for i in range(0, vals[0, :].size):
    plt.figure()
    plt.plot(range(10, vals[:, 0].size), stds[:, i])
    plt.plot(range(10, vals_old[:, 0].size), stds_old[:, i])
plt.show()
