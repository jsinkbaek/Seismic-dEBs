import numpy as np
from scipy.stats import sigmaclip
import matplotlib.pyplot as plt

period = 63.33

data_A = np.loadtxt('Data/processed/RV_results/rvA_not_8430105_4500_6700_100errors2_allvalues.txt')
data_B = np.loadtxt('Data/processed/RV_results/rvB_not_8430105_4500_6700_100errors2_allvalues.txt')

rvs_A = data_A[:, 2:]
rvs_B = data_B[:, 2:]

time_A = data_A[:, 0]
time_B = data_B[:, 0]

RV_A = np.empty(data_A[:, 0].shape)
RV_B = np.empty(data_B[:, 0].shape)
err_A = np.empty(data_A[:, 0].shape)
err_B = np.empty(data_B[:, 0].shape)

for i in range(0, rvs_A[:, 0].size):
    clipped, lower, upper = sigmaclip(rvs_A[i, :], low=4.0, high=4.0)
    RV_A[i] = np.mean(clipped)
    err_A[i] = np.std(clipped) / np.sqrt(clipped.size)

for i in range(0, rvs_B[:, 0].size):
    clipped, lower, upper = sigmaclip(rvs_B[i, :], low=4.0, high=4.0)
    RV_B[i] = np.mean(clipped)
    err_B[i] = np.std(clipped) / np.sqrt(clipped.size)

plt.errorbar(np.mod(time_A, period)/period, np.mean(rvs_A, axis=1), yerr=np.std(rvs_A, axis=1), fmt='b*')
plt.errorbar(np.mod(time_B, period)/period, np.mean(rvs_B, axis=1), yerr=np.std(rvs_B, axis=1), fmt='r*')
plt.xlabel('Orbital Phase')
plt.show(block=True)

save_data = np.empty((RV_A.size, 3))
save_data[:, 0] = time_A
save_data[:, 1] = RV_A
save_data[:, 2] = err_A
np.savetxt('Data/processed/RV_results/rvA_not_8430105_4500_6700_100errors2_clipped.txt', save_data)

save_data = np.empty((RV_B.size, 3))
save_data[:, 0] = time_B
save_data[:, 1] = RV_B
save_data[:, 2] = err_B
np.savetxt('Data/processed/RV_results/rvB_not_8430105_4500_6700_100errors2_clipped.txt', save_data)

