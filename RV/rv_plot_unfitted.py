import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.rcParams.update({'font.size': 25})
model_dir = '/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/NOT/kepler_LTF/'

model_filename = 'model.out'
# rva = 'rvA.out'
# rvb = 'rvB.out'
phase_model, rv_Am, rv_Bm = np.loadtxt(model_dir+model_filename, usecols=(0, 6, 7), unpack=True)
# rva = np.loadtxt(rva)
# rvb = np.loadtxt(rvb)

data_dir = '/home/sinkbaek/PycharmProjects/Seismic-dEBs/RV/Data/additionals/separation_routine/'
time_A, rv_A = np.loadtxt(data_dir+'4500_5825_rvA.txt', unpack=True)
time_B, rv_B, _ = np.loadtxt(data_dir+'4500_5825_rvB.txt', unpack=True)
index_B, _, mean_rv_B, error_B = np.loadtxt(data_dir+'refitted/mean_std_B.txt', unpack=True)
_, mean_rv_A, error_A = np.loadtxt(data_dir+'refitted/mean_std_A.txt', unpack=True)

time_A = time_A - (54998.2347431865 - 54976.6348)
time_B = time_B - (54998.2347431865 - 54976.6348)
print(time_B)

index_B = index_B.astype(int)
# print(index_B)

time_B = time_B[index_B]
rv_B = rv_B[index_B]

systemic_rv_estimate = 12.61
rv_A += systemic_rv_estimate
rv_B += systemic_rv_estimate
# mean_rv_A += systemic_rv_estimate
# mean_rv_B += systemic_rv_estimate

period = 63.327
phase_A = np.mod(time_A, period)/period
phase_B = np.mod(time_B, period)/period

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[2, 0])
ax1.set_xlabel('Orbital Phase')
ax1.set_ylabel('Radial Velocity (km/s)')
# ax2.set_ylabel('O - C')
ax1.set_xlim([0, 1.0])
# ax2.set_xlim([0, 1.0])

# a_mask = rva[:, 0] > 59000
# b_mask = rvb[:, 0] > 59000

phase_eclipse = 0.65892
approximate_hwidth= 0.02

ax1.errorbar(phase_A, rv_A, yerr=error_A, fmt='*', color='blue')
ax1.errorbar(phase_B, rv_B, yerr=error_B, fmt='*', color='red')
ax1.plot(phase_model, rv_Am, 'k--')
ax1.plot(phase_model, rv_Bm, 'k--')
ax1.fill_between([phase_eclipse-approximate_hwidth, phase_eclipse+approximate_hwidth], [-46.5, -46.5], [51, 51],
                 alpha=0.4, color='grey')
for i in range(0, rv_B.size):
    ax1.annotate(i, (phase_B[i], rv_B[i]), (phase_B[i], rv_B[i]+ (np.mod(i, rv_B.size//4)/(rv_B.size//4) - 0.5)*5),
                 arrowprops={'arrowstyle': '->'}, fontsize=15)
ax1.set_ylim([-46.5, 51])
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


save_array = np.empty((time_A.size, 3))
save_array[:, 0] = time_A + (54998.2347431865 - 54976.6348) + 54976.6348
save_array[:, 1] = rv_A
save_array[:, 2] = error_A
np.savetxt('rvA.dat', save_array)

save_array = np.empty((time_B.size, 3))
save_array[:, 0] = time_B + (54998.2347431865 - 54976.6348) + 54976.6348
save_array[:, 1] = rv_B
save_array[:, 2] = error_B
np.savetxt('rvB.dat', save_array)

