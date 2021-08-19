import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.rcParams.update({'font.size': 25})
os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/RV/Data/additionals/separation_routine/')

model_filename = '/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/kepler_LTF/model.out'
phase_model, rv_Bm, rv_Am = np.loadtxt(model_filename, usecols=(0, 6, 7), unpack=True)

time_A, rv_A = np.loadtxt('4500_6825_rvA.txt', unpack=True)
time_B, rv_B, _ = np.loadtxt('4500_6825_rvB.txt', unpack=True)
index_B, _, mean_rv_B, error_B = np.loadtxt('approx_500_width/mean_std_B.txt', unpack=True)
_, mean_rv_A, error_A = np.loadtxt('approx_500_width/mean_std_A.txt', unpack=True)

index_B = index_B.astype(int)
print(index_B)

time_B = time_B[index_B]
rv_B = rv_B[index_B]

systemic_rv_estimate = 12.61
rv_A += systemic_rv_estimate
rv_B += systemic_rv_estimate
mean_rv_A += systemic_rv_estimate
mean_rv_B += systemic_rv_estimate

period = 63.33
phase_A = np.mod(time_A, period)/period
phase_B = np.mod(time_B, period)/period

plt.figure(figsize=(16, 9))
plt.xlabel('Orbital Phase')
plt.ylabel('Radial Velocity (km/s)')
plt.errorbar(phase_A, rv_A, yerr=error_A, fmt='*', color='blue')
# plt.errorbar(phase_A, mean_rv_A, yerr=error_A, fmt='x', color='blue')
plt.errorbar(phase_B, rv_B, yerr=error_B, fmt='*', color='red')
# plt.errorbar(phase_B, mean_rv_B, yerr=error_B, fmt='x', color='red')
# plt.plot(phase_model, rv_Am - 3.443, 'k--')
# plt.plot(phase_model, rv_Bm - 3.443, 'k--')
plt.show()

