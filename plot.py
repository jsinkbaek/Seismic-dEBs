import importlib
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import MS_Teff_estimate as Tlib


def psd(x=None, y=None, loc=None):
    if isinstance(loc, str):
        data = np.loadtxt(loc)
        x, y = data[:, 0], data[:, 1]
    elif x and y is None:
        raise AttributeError("No input given to psd. Give either x and y, or a datafile location.")

    plt.figure()
    plt.plot(x, y, linewidth=0.5)
    plt.xlabel('Frequency muHz')
    plt.ylabel('Power Spectral Density')
    plt.show()


def lc(x=None, y=None, yerr=None, loc=None, period=None):
    if isinstance(loc, str):
        data = np.loadtxt(loc)
        x, y, yerr = data[:, 0], data[:, 1], data[:, 2]
    elif x and y is None:
        raise AttributeError("No input given to psd. Give either x and y, or a datafile location.")
    plt.figure()
    plt.errorbar(x, y, yerr, fmt='k.', markersize=0.5, elinewidth=0.5)
    plt.xlabel('BJD - 2400000')
    plt.ylabel('Relative Magnitude')
    plt.ylim([np.max(y+yerr)*1.1, -0.03])

    if period is not None:
        plt.figure()
        phase = np.mod(x, period)/period
        plt.errorbar(phase, y, yerr, fmt='k.', markersize=0.5, elinewidth=0.5)
        plt.xlabel('Phase')
        plt.ylabel('Relative Magnitude')
        plt.ylim([np.max(y + yerr) * 1.1, -0.003])
    plt.show()


def jktebop_model(loc_model, loc_lc, loc_lc_fit, loc_rvA, loc_rvB, period, initial_t):
    """
    Plot results from JKTEBOP model file against full lightcurves
    """
    matplotlib.rcParams.update({'font.size': 22})
    model = np.loadtxt(loc_model)
    lcurve = np.loadtxt(loc_lc)
    lcurve_fit = np.loadtxt(loc_lc_fit)
    rvA = np.loadtxt(loc_rvA)
    rvB = np.loadtxt(loc_rvB)

    phase_lc = np.mod(lcurve[:, 0] - initial_t, period) / period
    phase_lc_fit = np.mod(lcurve_fit[:, 0] - initial_t, period) / period
    phase_model = model[:, 0]
    phase_model_rv = model[:, 0]
    phase_rvA = np.mod(rvA[:, 0] - initial_t, period) / period
    phase_rvB = np.mod(rvB[:, 0] - initial_t, period) / period
    m = lcurve[:, 1]
    m_fit = lcurve_fit[:, 1]
    m_err = lcurve[:, 2]
    m_fit_err = lcurve_fit[:, 2]
    m_model = model[:, 1]
    rvA_model = model[:, 6]
    rvB_model = model[:, 7]

    # # Append to extend slightly beyong 0 and 1
    mask1 = phase_lc < 0.05
    mask3 = phase_model < 0.05
    mask5 = phase_lc_fit < 0.05
    phase_lc = np.append(phase_lc, phase_lc[mask1]+1.0)
    phase_model = np.append(phase_model, phase_model[mask3] + 1.0)
    phase_lc_fit = np.append(phase_lc_fit, phase_lc_fit[mask5] + 1.0)
    m = np.append(m, m[mask1])
    m_err = np.append(m_err, m_err[mask1])
    m_model = np.append(m_model, m_model[mask3])
    m_fit = np.append(m_fit, m_fit[mask5])
    m_fit_err = np.append(m_fit_err, m_fit_err[mask5])

    mask2 = (phase_lc > 0.95) & (phase_lc <= 1.0)
    mask4 = (phase_model > 0.95) & (phase_model <= 1.0)
    mask6 = (phase_lc_fit > 0.95) & (phase_lc_fit <= 1.0)
    phase_lc = np.append(phase_lc[mask2] - 1.0, phase_lc)
    phase_model = np.append(phase_model[mask4] - 1.0, phase_model)
    phase_lc_fit = np.append(phase_lc_fit[mask6] - 1.0, phase_lc_fit)
    m = np.append(m[mask2], m)
    m_err = np.append(m_err[mask2], m_err)
    m_model = np.append(m_model[mask4], m_model)
    m_fit = np.append(m_fit[mask6], m_fit)
    m_fit_err = np.append(m_fit_err[mask6], m_fit_err)

    # # Make plots # #
    plt.figure()
    plt.errorbar(phase_lc, m, m_err, fmt='*', color='gray', ecolor='gray', markersize=0.5, elinewidth=0.5, zorder=0)
    plt.errorbar(phase_lc_fit, m_fit, m_fit_err, fmt='k*', markersize=0.5, elinewidth=0.5, zorder=5)
    plt.plot(phase_model, m_model, '--', zorder=10)
    plt.xlabel('Phase')
    plt.ylabel('Relative magnitude')
    plt.ylim([np.max(lcurve[:, 1]*1.1), -0.006])
    plt.legend(['JKTEBOP fit', 'Data used for fit', 'Kepler Light Curve'])

    plt.figure()
    plt.errorbar(phase_rvA, rvA[:, 1], rvA[:, 2], fmt='*', markersize=7, elinewidth=3)
    plt.errorbar(phase_rvB, rvB[:, 1], rvB[:, 2], fmt='*', markersize=7, elinewidth=3)
    plt.plot(phase_model_rv, rvA_model, '--')
    plt.plot(phase_model_rv, rvB_model, '--')
    plt.xlabel('Phase')
    plt.ylabel('RV Signal [km/s]')
    plt.legend(['RV MS', 'RV RG', 'JKTEBOP RV MS', 'JKTEBOP RV RG'])
    plt.show()


def temperature_profile(T_MS, T_RG, R_MS, R_RG):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    lim_tess = Tlib.spectral_response_limits(Tlib.tess_spectral_response)
    wl_tess = np.linspace(lim_tess[0], lim_tess[1], 1000)
    srt = Tlib.tess_spectral_response(wl_tess)
    lim_kepler = Tlib.spectral_response_limits(Tlib.kepler_spectral_response)
    wl_kepler = np.linspace(lim_kepler[0], lim_kepler[1], 1000)
    srk = Tlib.kepler_spectral_response(wl_kepler)

    wl_planck = np.linspace(100, 14000, 1000)
    planck_rg = Tlib.planck_func(wl_planck, T_RG)
    planck_ms = Tlib.planck_func(wl_planck, T_MS)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(wl_planck, planck_rg, 'r')
    ax1.plot(wl_planck, planck_ms, 'y')
    ax1.set_ylabel('Radiance')
    ax1.set_title('Planck Curve')
    ax1.legend([r'$T_{RG}$ = ' + str(T_RG)+'K', r'$T_{MS}$ = ' + str(T_MS)+'K'])
    ax2.plot(wl_planck, planck_rg * R_RG ** 2, 'r')
    ax2.plot(wl_planck, planck_ms * R_MS ** 2, 'y')
    ax2.set_ylabel(r'Radiance $\cdot$ R²')
    ax2.legend([r'$T_{RG}$ = ' + str(T_RG) + 'K', r'$T_{MS}$ = ' + str(T_MS) + 'K'])
    ax2.set_xlabel('Wavelength [Å]')
    plt.xlim([3000, 12000])
    plt.show(block=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(wl_planck, planck_rg/np.max(planck_ms), 'r')
    ax1.plot(wl_planck, planck_ms/np.max(planck_ms), 'y')
    ax1.plot(wl_tess, srt, 'b')
    ax1.plot(wl_kepler, srk, color='darkorange')
    ax1.legend(['Scaled planck curve T='+str(T_RG), 'Scaled planck curve T='+str(T_MS),
                'TESS Spectral Response', 'Kepler Spectral Response'])
    ax1.set_ylabel('Normalized units')
    planck_tess_MS = Tlib.planck_func(wl_tess, T_MS)
    planck_tess_RG = Tlib.planck_func(wl_tess, T_RG)
    planck_kepler_MS = Tlib.planck_func(wl_kepler, T_MS)
    planck_kepler_RG = Tlib.planck_func(wl_kepler, T_RG)
    scale = np.max(planck_tess_MS*srt)
    ax2.plot(wl_tess, planck_tess_MS*srt/scale, 'b')
    ax2.plot(wl_tess, planck_tess_RG*srt/scale, 'b--')
    ax2.plot(wl_kepler, planck_kepler_MS*srk/scale, 'r')
    ax2.plot(wl_kepler, planck_kepler_RG*srk/scale, 'r--')
    ax2.legend([r'TESS T$_{MS}$='+str(T_MS), r'TESS T$_{RG}$='+str(T_RG), r'Kepler T$_{MS}$='+str(T_MS),
                r'Kepler T$_{RG}$='+str(T_RG)])
    ax2.set_title(r'Planck curve $\cdot$ Spectral response function')
    ax2.set_xlabel('Wavelength [Å]')
    ax2.set_ylabel('Normalized units')
    plt.xlim([3000, 12000])
    plt.show(block=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, )
    ax1.plot(wl_tess, planck_tess_MS*srt*(R_MS**2), 'b')
    ax2.plot(wl_tess, planck_tess_RG*srt*(R_RG**2), 'b')
    ax1.plot(wl_kepler, planck_kepler_MS*srk*(R_MS**2), 'r')
    ax2.plot(wl_kepler, planck_kepler_RG*srk*(R_RG**2), 'r')
    ax2.set_xlabel('Wavelength [Å]')
    plt.show(block=False)

    plt.figure()
    plt.plot(wl_tess, planck_tess_RG*srt*(R_RG**2)/(planck_tess_MS*srt*(R_MS**2)), 'b')
    plt.plot(wl_kepler, planck_kepler_RG*srk*(R_RG**2)/(planck_kepler_MS*srk*(R_MS**2)), 'r')
    plt.xlabel('Wavelength [Å]')
    plt.show()


# temperature_profile(5616, 5042, 0.727, 7.513)
jktebop_model('jktebop_kepler/model.out', 'lcmag_kepler_full.txt', 'lcmag_kepler.txt', 'jktebop_kepler/rvA.dat',
              'jktebop_kepler/rvB.dat', 63.32713, 54976.6351499878)
#jktebop_model('jktebop_tess/model.out', 'lcmag_tess_full.txt', 'jktebop_tess/rvA.dat',
#              'jktebop_tess/rvB.dat', 63.32713, 58712.9377353396)
# lc(loc='lcmag_kepler.txt', period=63.32713)
# psd(loc="datafiles/kasoc/8430105_psd.txt")
