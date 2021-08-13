from matplotlib import pyplot as plt; import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib

filename_1A = 'Data/processed/RV_results/rvA_not_8430105_4700_5400_100.txt'
filename_1B = 'Data/processed/RV_results/rvB_not_8430105_4700_5400_100.txt'
# filename_1A = 'Data/processed/RV_results/rvA_not_8430105_4500_6700_100errors2_clipped.txt'
# filename_1B = 'Data/processed/RV_results/rvB_not_8430105_4500_6700_100errors2_clipped.txt'
#filename_2A = 'Data/processed/RV_results/rvA_not_8430105_4700_5400_100_3errors2.txt'
#filename_2B = 'Data/processed/RV_results/rvB_not_8430105_4700_5400_100_3errors2.txt'
# filename_karsten_A = '../../../temp/8430105_RV_G_9.dat'
# filename_karsten_B = '../../../temp/8430105_RV_MS_9.dat'
filename_karsten_A = 'Data/processed/RV_results/rvA_not_8430105_4700_5400_100_ignoreB.txt'
filename_karsten_B = 'Data/processed/RV_results/rvB_not_8430105_4700_5400_100_ignoreB.txt'
model_filename = '../Binary_Analysis/JKTEBOP/kepler_LTF/model.out'

times_1A, rv_1A, err_1A = np.loadtxt(filename_1A, unpack=True)
times_1B, rv_1B, err_1B = np.loadtxt(filename_1B, unpack=True)
#times_2A, rv_2A, err_2A = np.loadtxt(filename_2A, unpack=True)
#times_2B, rv_2B, err_2B = np.loadtxt(filename_2B, unpack=True)
times_kA, rv_kA, err_kA = np.loadtxt(filename_karsten_A, unpack=True)
times_kB, rv_kB, err_kB = np.loadtxt(filename_karsten_B, unpack=True)

period = 63.33
system_rv = 16.053

times_1A -= 2400000 + 54976.6348
times_1B -= 2400000 + 54976.6348
#times_2A -= 2400000 + 54976.6348
#times_2B -= 2400000 + 54976.6348
times_kA -= 2400000 + 54976.6348
times_kB -= 2400000 + 54976.6348

phase_model, rv_Bm, rv_Am = np.loadtxt(model_filename, usecols=(0, 6, 7), unpack=True)
interp_Am = interp1d(phase_model, rv_Am)
phase_1A = np.mod(times_1A, period)/period
phase_1B = np.mod(times_1B, period)/period
rvA_model_interp_vals =  interp_Am(phase_1A)


def rv_eval_plus_constant(rv_constant):
    return np.sum(np.abs(rv_1A - (rvA_model_interp_vals + rv_constant)))


system_rv_new = -minimize(rv_eval_plus_constant, x0=np.array([0])).x
print(system_rv_new)

plt.figure(figsize=(16, 9))
#plt.errorbar(np.mod(times_2A, period)/period, rv_2A, yerr=err_2A, fmt='g*')
#plt.errorbar(np.mod(times_2B, period)/period, rv_2B, yerr=err_2B, fmt='y*')
plt.errorbar(np.mod(times_1A, period)/period, rv_1A-system_rv, yerr=err_1A, fmt='b*')
plt.errorbar(np.mod(times_1B, period)/period, rv_1B-system_rv, yerr=err_1B, fmt='r*')
plt.errorbar(np.mod(times_kA, period)/period, rv_kA-system_rv, yerr=err_kA, fmt='g*')
plt.errorbar(np.mod(times_kB, period)/period, rv_kB-system_rv, yerr=err_kB, fmt='y*')
#plt.legend(['Component A errors by full split', 'Component B', 'Component A errors retain sep spectrum', 'Component B'])
plt.legend(['Component A', 'Component B', 'Component A ignoreB', 'Component B'])
plt.plot(phase_model, rv_Am-system_rv-system_rv_new, 'k-')
plt.plot(phase_model, rv_Bm-system_rv-system_rv_new, 'k-')
plt.xlabel('Orbital Phase')
plt.ylabel(f'Radial Velocity - {system_rv_new} [km/s]')
plt.show(block=True)

