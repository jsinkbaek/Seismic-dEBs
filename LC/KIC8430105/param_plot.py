import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

matplotlib.rcParams.update({'font.size': 18})
# TESS, TESS LTF, TESS 3rd light, kepler LTF, KASOC (fitted) (task 9 results, and 1 sigma errors)
#       KASOC (fitted) uses LTF errors atm, as task 9 is not finished for it
mA = np.array([
    0.8273099647,
    0.8273521162,
    0.8271298638,
    0.8275945275,
    0.8275966467
])
mB = np.array([
    1.3133116838,
    1.3132801445,
    1.3129389222,
    1.3139775147,
    1.3141476298
])
rA = np.array([
    0.7462891906,
    0.7576974571,
    0.7681889901,
    0.7691084114,
    0.7621412720
])
rB = np.array([
    7.5930451879,
    7.5247775955,
    7.5223396909,
    7.6736902033,
    7.6923684363
])

# Errors
mA_err = np.array([
    0.0137023797,
    0.0136525927,
    0.0137714714,
    0.0137987820,
    0.0136116558
])
mB_err = np.array([
    0.0177221276,
    0.0178754996,
    0.0175323601,
    0.0178136540,
    0.0175913395
])
rA_err = np.array([
    0.0143775457,
    0.0112818365,
    0.0044852035,
    0.0056814276,
    0.0054131390
])
rB_err = np.array([
    0.1224946421,
    0.0820654610,
    0.0405367736,
    0.0490668428,
    0.0471883768
])

# mass A, mass B, radius A, radius B
tess       =    np.array([mA[0], mB[0], rA[0], rB[0]])
tess_ltf        =    np.array([mA[1], mB[1], rA[1], rB[1]])
tess3        =    np.array([mA[2], mB[2], rA[2], rB[2]])
ltf     =    np.array([mA[3], mB[3], rA[3], rB[3]])
kasfit     =    np.array([mA[4], mB[4], rA[4], rB[4]])
tess_err   =    np.array([mA_err[0], mB_err[0], rA_err[0], rB_err[0]])
tess_ltf_err    =    np.array([mA_err[1], mB_err[1], rA_err[1], rB_err[1]])
tess3_err    =    np.array([mA_err[2], mB_err[2], rA_err[2], rB_err[2]])
ltf_err =    np.array([mA_err[3], mB_err[3], rA_err[3], rB_err[3]])
kasfit_err =    np.array([mA_err[4], mB_err[4], rA_err[4], rB_err[4]])


plt.figure()
plt.errorbar(tess[0], tess[2], yerr=tess_err[2], xerr=tess_err[0], fmt='r*', markersize=10)
plt.errorbar(tess_ltf[0], tess_ltf[2], yerr=tess_ltf_err[2], xerr=tess_ltf_err[0], fmt='d', ecolor='chocolate',
             markersize=10)
plt.errorbar(tess3[0], tess3[2], yerr=tess3_err[2], xerr=tess3_err[0], fmt='d', ecolor='indianred', markersize=10)
plt.errorbar(ltf[0], ltf[2], yerr=ltf_err[2], xerr=ltf_err[0], fmt='b*', markersize=10)
plt.errorbar(kasfit[0], kasfit[2], yerr=kasfit_err[2], xerr=kasfit_err[0], fmt='m*', markersize=10)
plt.ylabel(r'Radius $[R_\bigodot]$')
plt.xlabel(r'Mass $[M_\bigodot]$')
plt.legend(['TESS MS', 'TESS LTF MS', 'TESS 3rd light MS', 'Kepler LTF MS', 'Kepler KASOC MS'])
plt.show(block=False)

plt.figure()
plt.errorbar(tess[1], tess[3], yerr=tess_err[3], xerr=tess_err[1], fmt='rH', markersize=10)
plt.errorbar(tess_ltf[1], tess_ltf[3], yerr=tess_ltf_err[3], xerr=tess_ltf_err[1], fmt='d', ecolor='chocolate', markersize=10)
plt.errorbar(tess3[1], tess3[3], yerr=tess3_err[3], xerr=tess3_err[1], fmt='d', ecolor='indianred', markersize=10)
plt.errorbar(ltf[1], ltf[3], yerr=ltf_err[3], xerr=ltf_err[1], fmt='bH', markersize=10)
plt.errorbar(kasfit[1], kasfit[3], yerr=kasfit_err[3], xerr=kasfit_err[1], fmt='mH', markersize=10)
plt.ylabel(r'Radius $[R_\bigodot]$')
plt.xlabel(r'Mass $[M_\bigodot]$')
plt.legend(['TESS RG', 'TESS LTF RG', 'TESS 3rd light RG', 'Kepler LTF RG', 'Kepler KASOC RG'])
plt.show(block=False)

plt.figure()
markersize=5
plt.errorbar(tess[0], tess[2], yerr=tess_err[2], xerr=tess_err[0], fmt='r*', markersize=markersize)
plt.errorbar(tess3[0], tess3[2], yerr=tess3_err[2], xerr=tess3_err[0], fmt='d', ecolor='indianred', markersize=markersize)
plt.errorbar(ltf[0], ltf[2], yerr=ltf_err[2], xerr=ltf_err[0], fmt='b*', markersize=markersize)
plt.errorbar(kasfit[0], kasfit[2], yerr=kasfit_err[2], xerr=kasfit_err[0], fmt='m*', markersize=markersize)
plt.errorbar(tess[1], tess[3], yerr=tess_err[3], xerr=tess_err[1], fmt='rH', markersize=markersize)
plt.errorbar(tess3[1], tess3[3], yerr=tess3_err[3], xerr=tess3_err[1], fmt='d', ecolor='indianred', markersize=markersize)
plt.errorbar(ltf[1], ltf[3], yerr=ltf_err[3], xerr=ltf_err[1], fmt='bH', markersize=markersize)
plt.errorbar(kasfit[1], kasfit[3], yerr=kasfit_err[3], xerr=kasfit_err[1], fmt='mH', markersize=markersize)
plt.ylabel(r'Radius $[R_\bigodot]$')
plt.xlabel(r'Mass $[M_\bigodot]$')
plt.legend(['TESS MS', 'TESS 3rd light MS', 'Kepler LTF MS', 'Kepler KASOC MS', 'TESS RG', 'TESS 3rd light RG',
            'Kepler LTF RG', 'Kepler KASOC RG'])
plt.show()
