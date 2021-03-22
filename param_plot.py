import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

matplotlib.rcParams.update({'font.size': 18})
# TESS, LTF, KASOC (fitted) (task 9 results, and 1 sigma errors)
#       KASOC (fitted) uses LTF errors atm, as task 9 is not finished for it
mA = np.array([
    0.8273099647,
    0.8275970054,
    0.8275643013
])
mB = np.array([
    1.3133116838,
    1.3139827531,
    1.3140757635
])
rA = np.array([
    0.7462891906,
    0.7691149602,
    0.7621735195
])
rB = np.array([
    7.5930451879,
    7.6737440761,
    7.6911705511
])

# Errors
mA_err = np.array([
    0.0137023797,
    0.0137931427,
    0.0137931427
])
mB_err = np.array([
    0.0177221276,
    0.0178204628,
    0.0178204628
])
rA_err = np.array([
    0.0143775457,
    0.0056879327,
    0.0056879327
])
rB_err = np.array([
    0.1224946421,
    0.0489342509,
    0.0489342509
])

# mass A, mass B, radius A, radius B
tess       =    np.array([mA[0], mB[0], rA[0], rB[0]])
ltf        =    np.array([mA[1], mB[1], rA[1], rB[1]])
kasfit     =    np.array([mA[2], mB[2], rA[2], rB[2]])
tess_err   =    np.array([mA_err[0], mB_err[0], rA_err[0], rB_err[0]])
ltf_err    =    np.array([mA_err[1], mB_err[1], rA_err[1], rB_err[1]])
kasfit_err =    np.array([mA_err[2], mB_err[2], rA_err[2], rB_err[2]])


plt.figure()
plt.errorbar(tess[0], tess[2], yerr=tess_err[2], xerr=tess_err[0], fmt='r*', markersize=10)
plt.errorbar(ltf[0], ltf[2], yerr=ltf_err[2], xerr=ltf_err[0], fmt='b*', markersize=10)
plt.errorbar(kasfit[0], kasfit[2], yerr=kasfit_err[2], xerr=kasfit_err[0], fmt='m*', markersize=10)
plt.ylabel(r'Radius $[R_\bigodot]$')
plt.xlabel(r'Mass $[M_\bigodot]$')
plt.legend(['TESS MS', 'Kepler LTF MS', 'Kepler KASOC MS'])
plt.show(block=False)

plt.figure()
plt.errorbar(tess[1], tess[3], yerr=tess_err[3], xerr=tess_err[1], fmt='rH', markersize=10)
plt.errorbar(ltf[1], ltf[3], yerr=ltf_err[3], xerr=ltf_err[1], fmt='bH', markersize=10)
plt.errorbar(kasfit[1], kasfit[3], yerr=kasfit_err[3], xerr=kasfit_err[1], fmt='mH', markersize=10)
plt.ylabel(r'Radius $[R_\bigodot]$')
plt.xlabel(r'Mass $[M_\bigodot]$')
plt.legend(['TESS RG', 'Kepler LTF RG', 'Kepler KASOC RG'])
plt.show(block=False)

plt.figure()
markersize=5
plt.errorbar(tess[0], tess[2], yerr=tess_err[2], xerr=tess_err[0], fmt='r*', markersize=markersize)
plt.errorbar(ltf[0], ltf[2], yerr=ltf_err[2], xerr=ltf_err[0], fmt='b*', markersize=markersize)
plt.errorbar(kasfit[0], kasfit[2], yerr=kasfit_err[2], xerr=kasfit_err[0], fmt='m*', markersize=markersize)
plt.errorbar(tess[1], tess[3], yerr=tess_err[3], xerr=tess_err[1], fmt='rH', markersize=markersize)
plt.errorbar(ltf[1], ltf[3], yerr=ltf_err[3], xerr=ltf_err[1], fmt='bH', markersize=markersize)
plt.errorbar(kasfit[1], kasfit[3], yerr=kasfit_err[3], xerr=kasfit_err[1], fmt='mH', markersize=markersize)
plt.ylabel(r'Radius $[R_\bigodot]$')
plt.xlabel(r'Mass $[M_\bigodot]$')
plt.legend(['TESS MS', 'Kepler LTF MS', 'Kepler KASOC MS', 'TESS RG', 'Kepler LTF RG', 'Kepler KASOC RG'])
plt.show()
