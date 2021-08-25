import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.rcParams.update({'font.size': 25})
os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/RV/Data/additionals/separation_routine/')

time_A, rv_A = np.loadtxt('4500_6825_rvA.txt', unpack=True)
time_B, rv_B, _ = np.loadtxt('4500_6825_rvB.txt', unpack=True)
index_B, _, mean_rv_B, error_B = np.loadtxt('approx_500_width/mean_std_B.txt', unpack=True)
_, mean_rv_A, error_A = np.loadtxt('approx_500_width/mean_std_A.txt', unpack=True)

index_B = index_B.astype(int)

time_B = time_B[index_B]
rv_B = rv_B[index_B]

systemic_rv_estimate = 12.61
rv_A += systemic_rv_estimate
rv_B += systemic_rv_estimate
mean_rv_A += systemic_rv_estimate
mean_rv_B += systemic_rv_estimate

save_array = np.empty((time_A.size, 3))
save_array[:, 0] = time_A + 54976.6348
save_array[:, 1] = mean_rv_A
save_array[:, 2] = error_A
np.savetxt('rvA_mean.dat', save_array)

save_array = np.empty((time_B.size, 3))
save_array[:, 0] = time_B + 54976.6348
save_array[:, 1] = mean_rv_B
save_array[:, 2] = error_B
np.savetxt('rvB_mean.dat', save_array)

