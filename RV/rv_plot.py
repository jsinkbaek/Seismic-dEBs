import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.rcParams.update({'font.size': 25})
os.chdir('/home/sinkbaek/PycharmProjects/Seismic-dEBs/RV/Data/additionals/separation_routine/')

model_filename = '/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/NOT/kepler_LTF/model.out'
rva = '/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/NOT/kepler_LTF/rvA.out'
rvb = '/home/sinkbaek/PycharmProjects/Seismic-dEBs/Binary_Analysis/JKTEBOP/NOT/kepler_LTF/rvB.out'
phase_model, rv_Am, rv_Bm = np.loadtxt(model_filename, usecols=(0, 6, 7), unpack=True)
rva = np.loadtxt(rva)
rvb = np.loadtxt(rvb)

# time_A, rv_A = np.loadtxt('4500_6825_rvA.txt', unpack=True)
# time_B, rv_B, _ = np.loadtxt('4500_6825_rvB.txt', unpack=True)
# index_B, _, mean_rv_B, error_B = np.loadtxt('approx_500_width/mean_std_B.txt', unpack=True)
# _, mean_rv_A, error_A = np.loadtxt('approx_500_width/mean_std_A.txt', unpack=True)

# time_A = time_A - (54998.2347431865 - 54976.6348)
# time_B = time_B - (54998.2347431865 - 54976.6348)

# index_B = index_B.astype(int)
# print(index_B)

# time_B = time_B[index_B]
# rv_B = rv_B[index_B]

# systemic_rv_estimate = 12.61
# rv_A += systemic_rv_estimate
# rv_B += systemic_rv_estimate
# mean_rv_A += systemic_rv_estimate
# mean_rv_B += systemic_rv_estimate

# period = 63.33
# phase_A = np.mod(time_A, period)/period
# phase_B = np.mod(time_B, period)/period

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(3, 1)
ax1 = fig.add_subplot(gs[0:2, 0])
ax2 = fig.add_subplot(gs[2, 0])
ax2.set_xlabel('Orbital Phase')
ax1.set_ylabel('Radial Velocity (km/s)')
ax2.set_ylabel('O - C')
ax1.set_xlim([0, 1.0])
ax2.set_xlim([0, 1.0])
print(rva)

a_mask = rva[:, 0] > 59000
b_mask = rvb[:, 0] > 59000

ax1.errorbar(rva[:, 3], rva[:, 1], yerr=rva[:, 2], fmt='*', color='blue')
ax1.errorbar(rvb[:, 3], rvb[:, 1], yerr=rvb[:, 2], fmt='*', color='red')
ax1.plot(phase_model, rv_Am, 'k--')
ax1.plot(phase_model, rv_Bm, 'k--')
ax1.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)

ax2.errorbar(rva[:, 3], rva[:, 5], yerr=rva[:, 2], fmt='*', color='blue')
# ax2.errorbar(rva[a_mask, 3], rva[a_mask, 5], yerr=rva[a_mask, 2], fmt='*', color='green')
ax2.errorbar(rvb[:, 3], rvb[:, 5], yerr=rvb[:, 2], fmt='*', color='red')
# ax2.errorbar(rvb[b_mask, 3], rvb[b_mask, 5], yerr=rvb[b_mask, 2], fmt='*', color='magenta')
ax2.plot([0, 1], [0, 0], '--', color='grey')
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# save_array = np.empty((time_A.size, 3))
# save_array[:, 0] = time_A + 54976.6348
# save_array[:, 1] = rv_A
# save_array[:, 2] = error_A
# np.savetxt('rvA.dat', save_array)

# save_array = np.empty((time_B.size, 3))
# save_array[:, 0] = time_B + 54976.6348
# save_array[:, 1] = rv_B
# save_array[:, 2] = error_B
# np.savetxt('rvB.dat', save_array)

