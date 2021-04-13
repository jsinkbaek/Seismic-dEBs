import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from LC import MS_Teff_estimate as Tlib
from scipy.interpolate import interp1d


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


def lc(x=None, y=None, yerr=None, loc=None, period=None, model_loc=None, phase_xlim=None):
    matplotlib.rcParams.update({'font.size': 18})
    if isinstance(loc, str):
        data = np.loadtxt(loc)
        x, y, yerr = data[:, 0] - 58712.9380709522, data[:, 1], data[:, 2]
    elif x and y is None:
        raise AttributeError("No input given to psd. Give either x and y, or a datafile location.")
    plt.figure()
    plt.errorbar(x, y, yerr, fmt='k.', markersize=0.5, elinewidth=0.5)
    plt.xlabel('BJD - 2400000')
    plt.ylabel('Relative Magnitude')
    plt.ylim([np.max(y+yerr)*1.1, -0.007])

    if period is not None:
        plt.figure()
        phase = np.mod(x, period)/period
        plt.errorbar(phase, y, yerr, fmt='.', color='gray', markersize=0.5, elinewidth=0.5, errorevery=10)
        plt.errorbar(phase-1, y, yerr, fmt='.', color='gray', markersize=0.5, elinewidth=0.5, errorevery=10)
        plt.errorbar(phase+1, y, yerr, fmt='.', color='gray', markersize=0.5, elinewidth=0.5, errorevery=10)
        plt.xlabel('Phase')
        plt.ylabel('Relative Magnitude')
        plt.ylim([np.max(y + yerr) * 1.1, -0.007])
        if isinstance(phase_xlim, list):
            plt.xlim(phase_xlim)
        else:
            plt.xlim([-0.05, 1.05])
        if isinstance(model_loc, str):
            model_data = np.loadtxt(model_loc)
            model_phase = model_data[:, 0]
            model_mag = model_data[:, 1]
            plt.plot(model_phase, model_mag, 'k--')
            plt.plot(model_phase-1, model_mag, 'k--')
            plt.plot(model_phase+1, model_mag, 'k--')

    plt.show()


def lc_plot2(loc1, loc2, fmt1='b.', fmt2='r.', xlabel='BJD - 2400000', ylabel='Relative Magnitude', xlim=None,
             ylim=None, legend=None):

    data1 = np.loadtxt(loc1)
    tm1, fl1, fl1_err = data1[:, 0], data1[:, 1], data1[:, 2]
    data2 = np.loadtxt(loc2)

    tm2, fl2, fl2_err = data2[:, 0], data2[:, 1], data2[:, 2]
    tm2_u, u_idx = np.unique(tm2, return_index=True)
    plt.figure()
    plt.plot(tm1, fl1, fmt1, markersize=1)
    plt.plot(tm2_u, fl2[u_idx], fmt2, markersize=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if legend is not None:
        plt.legend(legend)
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

    plt.figure()
    plt.plot(wl_planck, planck_rg/np.max(planck_rg))
    plt.xlabel('Wavelength [Å]')
    plt.ylabel('Normalized units')
    plt.title('Planck Curve and Kepler spectral response function')
    plt.plot(wl_kepler, srk, 'r')
    plt.legend(['Radiance of Planck Curve (T=5042K)', 'Spectral Response Function of Kepler CCD'])
    plt.show()

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


def obs_v_cal(loc_lc1, loc_model1, loc_lc2=None, loc_model2=None, t0=54976.635, period=63.32713,
              phase_lim_primary=np.array([-0.023, 0.177-0.156]), phase_lim_secondary=np.array([0.473-0.156, 0.522-0.156])):
    data_lc1 = np.loadtxt(loc_lc1)
    time_lc1, mag_lc1 = data_lc1[:, 0], data_lc1[:, 1]

    data_model1 = np.loadtxt(loc_model1)
    phase_model1, mag_model1 = data_model1[:, 0], data_model1[:, 1]
    phase_lc1 = np.mod(time_lc1, period)/period
    cut_mask_1 = np.diff(phase_lc1) < -0.002
    cut_idx_1 = np.where(cut_mask_1)[0]
    time_lc1_split = np.split(time_lc1, cut_idx_1)
    mag_lc1_split = np.split(mag_lc1, cut_idx_1)

    time_lc2, mag_lc2, phase_model2, mag_model2 = None, None, None, None
    if loc_lc2 and loc_model2 is not None:
        data_lc2 = np.loadtxt(loc_lc2)
        time_lc2, mag_lc2 = data_lc2[:, 0], data_lc2[:, 1]
        data_model2 = np.loadtxt(loc_model2)
        phase_model2, mag_model2 = data_model2[:, 0], data_model2[:, 1]
        phase_lc2 = np.mod(time_lc2, period)/period
        cut_mask_2 = np.diff(phase_lc2) < -0.002
        cut_idx_2 = np.where(cut_mask_2)[0]
        time_lc2_split = np.split(time_lc2, cut_idx_2)
        mag_lc2_split = np.split(mag_lc2, cut_idx_2)

    # # # Plot Model Comparison # # #
    nrows = 6
    for i in range(0, int(np.ceil(len(time_lc1_split)/nrows))):
        fig, axs = plt.subplots(nrows=nrows, ncols=2)
        j = 0
        for k in range(i*nrows, i*nrows + nrows):
            try:
                time_lc1_split[k]
                if loc_lc2 and loc_model2 is not None:
                    time_lc2_split[k]
            except IndexError:
                print('Out of bounds')
                break
            ax1, ax2 = axs[j, 0], axs[j, 1]
            ax1.plot(time_lc1_split[k], mag_lc1_split[k], 'r.', markersize=1.5)
            ax2.plot(time_lc1_split[k], mag_lc1_split[k], 'r.', markersize=1.5)
            ax1.plot(t0 + period * phase_model1 + period*k, mag_model1, 'k--', linewidth=1)
            ax1.plot(t0 - period * phase_model1 + period*k, mag_model1[::-1], 'k--', label='_nolegend_', linewidth=1)
            ax2.plot(t0 + period * phase_model1 + period*k, mag_model1, 'k--', linewidth=1)
            ax2.plot(t0 - period * phase_model1 + period*k, mag_model1[::-1, ], 'k--', label='_nolegend_', linewidth=1)
            if loc_lc2 and loc_model2 is not None:
                ax1.plot(time_lc2_split[k], mag_lc2_split[k], 'b.', markersize=1.5)
                ax2.plot(time_lc2_split[k], mag_lc2_split[k], 'b.', markersize=1.5)
                ax1.plot(t0 + period * phase_model2 + period*k, mag_model2, 'k-', linewidth=1)
                ax1.plot(t0 - period * phase_model2 + period * k, mag_model2[::-1], 'k-', label='_nolegend_',
                         linewidth=1)
                ax2.plot(t0 + period * phase_model2 + period*k, mag_model2, 'k-', linewidth=1)
                ax2.plot(t0 - period * phase_model2 + period * k, mag_model2[::-1, ], 'k-', label='_nolegend_',
                         linewidth=1)
                if j==0:
                    ax2.legend(['Light Curve 1', 'Light Curve 1 Model', 'Light Curve 2', 'Light Curve 2 Model'])
            else:
                if j==0:
                    ax2.legend(['Light Curve', 'Light Curve Model'])
            ax1.set_xlim(t0 + period * phase_lim_primary + period*k)
            ax2.set_xlim(t0 + period * phase_lim_secondary + period*k)
            j+=1
        else:
            plt.show(block=False)
            continue
        plt.show(block=False)
        break
    plt.show(block=False)

    # # # Plot O-C # # #
    nrows = 6
    mag_itp_dat1 = np.append(mag_model1[::-1], mag_model1)
    if loc_lc2 and loc_model2 is not None:
        mag_itp_dat2 = np.append(mag_model2[::-1], mag_model2)
    for i in range(0, int(np.ceil(len(time_lc1_split) / nrows))):
        fig, axs = plt.subplots(nrows=nrows, ncols=2, sharey='all')
        j = 0
        for k in range(i * nrows, i * nrows + nrows):
            try:
                time_lc1_split[k]
                if loc_lc2 and loc_model2 is not None:
                    time_lc2_split[k]
            except IndexError:
                print('Out of bounds')
                break
            ax1, ax2 = axs[j, 0], axs[j, 1]
            t_itp_dat1 = np.append(t0 - period*phase_model1 + period*k, t0 + period*phase_model1 + period*k)
            f_itp_1 = interp1d(t_itp_dat1, mag_itp_dat1)
            ax1.plot(time_lc1_split[k], mag_lc1_split[k] - f_itp_1(time_lc1_split[k]), 'r.', markersize=1.5)
            ax1.plot(t_itp_dat1, np.zeros(t_itp_dat1.shape), 'k--', linewidth=1, label='_nolegend_')
            ax2.plot(time_lc1_split[k], mag_lc1_split[k] - f_itp_1(time_lc1_split[k]), 'r.', markersize=1.5)
            ax2.plot(t_itp_dat1, np.zeros(t_itp_dat1.shape), 'k--', linewidth=1, label='_nolegend_')

            if loc_lc2 and loc_model2 is not None:
                t_itp_dat2 = np.append(t0 - period*phase_model2 + period*k, t0 + period*phase_model2 + period*k)
                f_itp_2 = interp1d(t_itp_dat2, mag_itp_dat2)
                ax1.plot(time_lc2_split[k], mag_lc2_split[k] - f_itp_2(time_lc2_split[k]), 'b.', markersize=1.5)
                ax2.plot(time_lc2_split[k], mag_lc2_split[k] - f_itp_2(time_lc2_split[k]), 'b.', markersize=1.5)

                if j == 0:
                    ax2.legend(['O-C Light Curve 1', 'O-C Light Curve 2'])
            else:
                if j == 0:
                    ax2.legend(['O-C Light Curve'])
            ax1.set_xlim(t0 + period * phase_lim_primary + period * k)
            ax2.set_xlim(t0 + period * phase_lim_secondary + period * k)
            ax1.set_ylim([-0.002, 0.002])
            ax2.set_ylim([-0.002, 0.002])
            j += 1
        else:
            plt.show(block=False)
            continue
        plt.show(block=False)
        break
    plt.show(block=True)


# noinspection PyUnboundLocalVariable
def obs_v_cal_folded(loc_lc1, loc_model1, loc_lc2=None, loc_model2=None, t0=54976.635, period=63.32713, legend=None,
                     phase_lim_primary=np.array([-0.0205, 0.01915]),
                     phase_lim_secondary=np.array([0.341355-0.0219, 0.341355+0.0218]), label1='Light Curve 1',
                     labelm1='Model 1', label2='Light Curve 2', labelm2='Model 2', o_c_ylim=None, plot_std=True,
                     marker1='y.', marker2='c.', line1='k--', line2='m-.', errorbar=True, color1='k', color2='m'):
    matplotlib.rcParams.update({'font.size': 17})

    data_lc1 = np.loadtxt(loc_lc1)
    time_lc1, mag_lc1, err_lc1 = data_lc1[:, 0], data_lc1[:, 1], data_lc1[:, 2]
    data_model1 = np.loadtxt(loc_model1)
    phase_model1, mag_model1 = data_model1[:, 0], data_model1[:, 1]
    phase_lc1 = np.mod(time_lc1-t0, period) / period

    if loc_lc2 and loc_model2 is not None:
        data_lc2 = np.loadtxt(loc_lc2)
        time_lc2, mag_lc2, err_lc2 = data_lc2[:, 0], data_lc2[:, 1], data_lc2[:, 2]
        data_model2 = np.loadtxt(loc_model2)
        phase_model2, mag_model2 = data_model2[:, 0], data_model2[:, 1]
        phase_lc2 = np.mod(time_lc2-t0, period)/period

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.plot(phase_lc1, mag_lc1, 'y.', markersize=1)
    ax1.plot(phase_lc1-1, mag_lc1, 'y.', markersize=1, label='_nolegend_')
    ax1.plot(phase_model1, mag_model1, 'k--', linewidth=2)
    ax1.plot(phase_model1 - 1, mag_model1, 'k--', linewidth=2, label='_nolegend_')
    ax1.set_xlim(phase_lim_primary)
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Relative Magnitude')
    if loc_lc2 and loc_model2 is not None:
        ax1.plot(phase_lc2, mag_lc2, 'c.', markersize=1)
        ax1.plot(phase_lc2 - 1, mag_lc2, 'c.', markersize=1, label='_nolegend_')
        ax1.plot(phase_model2, mag_model2, 'm-.', linewidth=2)
        ax1.plot(phase_model2-1, mag_model2, 'm-.', linewidth=2, label='_nolegend_')

    ax2.plot(phase_lc1, mag_lc1, 'y.', markersize=1)
    ax2.plot(phase_model1, mag_model1, 'k--', linewidth=2)
    ax2.set_xlim(phase_lim_secondary)
    ax2.set_xlabel('Phase')
    if loc_lc2 and loc_model2 is not None:
        ax2.plot(phase_lc2, mag_lc2, 'c.', markersize=1)
        ax2.plot(phase_model2, mag_model2, 'm-.', linewidth=2)
        ax2.legend(['Light Curve 1', 'Model 1', 'Light Curve 2', 'Model 2'], markerscale=8)
    else:
        ax2.legend(['Light Curve', 'Model'], markerscale=8)
    if legend is not None:
        ax2.legend(legend, markerscale=8)

    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax2.set_ylim(ax2.get_ylim()[::-1])

    plt.show(block=False)

    if loc_lc2 and loc_model2 is not None:
        fig, axs = plt.subplots(nrows=2, ncols=2, sharey='all')
        ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

        ax1.plot(phase_lc1, mag_lc1, 'y.', markersize=1, label='_nolegend_')
        ax1.plot(phase_lc1 - 1, mag_lc1, 'y.', markersize=1, label='_nolegend_')
        ax1.plot(phase_model1, mag_model1, 'k--', linewidth=2, label='_nolegend_')
        ax1.plot(phase_model1 - 1, mag_model1, 'k--', linewidth=2, label='_nolegend_')
        ax1.set_xlim(phase_lim_primary)
        ax1.set_ylabel('Relative Magnitude')

        ax2.plot(phase_lc2, mag_lc2, 'c.', markersize=1, label=label1)
        ax2.plot(phase_lc2 - 1, mag_lc2, 'c.', markersize=1, label='_nolegend_')
        ax2.plot(phase_model2, mag_model2, 'm-.', linewidth=2, label=labelm1)
        ax2.plot(phase_model2 - 1, mag_model2, 'm-.', linewidth=2, label='_nolegend_')
        ax2.set_xlim(phase_lim_primary)

        ax3.plot(phase_lc1, mag_lc1, 'y.', markersize=1, label=label2)
        ax3.plot(phase_model1, mag_model1, 'k--', linewidth=2,  label=labelm2)
        ax3.set_xlim(phase_lim_secondary)
        ax3.set_ylabel('Relative Magnitude')
        ax3.set_xlabel('Phase')

        ax4.plot(phase_lc2, mag_lc2, 'c.', markersize=1, label='_nolegend_')
        ax4.plot(phase_model2, mag_model2, 'm-.', linewidth=2, label='_nolegend_')
        ax4.set_xlim(phase_lim_secondary)
        ax4.set_xlabel('Phase')

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

        fig.legend(lines, labels, markerscale=8)
        ax1.set_ylim(ax1.get_ylim()[::-1])
        plt.show(block=False)

    # # # O - C plot # # #

    model1_interp = interp1d(phase_model1, mag_model1)
    o_c_1 = mag_lc1 - model1_interp(phase_lc1)
    if loc_lc2 and loc_model2 is not None:
        model2_interp = interp1d(phase_model2, mag_model2)
        o_c_2 = mag_lc2 - model2_interp(phase_lc2)

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0,
                                                                                       'height_ratios': [2, 1]})
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    print(mag_lc1.shape, err_lc1.shape)
    if errorbar:
        ax1.errorbar(phase_lc1, mag_lc1, err_lc1, fmt=marker1, markersize=1, errorevery=10)
        ax1.errorbar(phase_lc1-1, mag_lc1, err_lc1, fmt=marker1, markersize=1, errorevery=10, label='_nolegend_')
    else:
        ax1.plot(phase_lc1, mag_lc1, marker1, markersize=1)
        ax1.plot(phase_lc1 - 1, mag_lc1, marker1, markersize=1, label='_nolegend_')
    ax1.plot(phase_model1, mag_model1, line1, linewidth=2)
    ax1.plot(phase_model1 - 1, mag_model1, line1, linewidth=2, label='_nolegend_')
    ax1.set_xlim(phase_lim_primary)
    if loc_lc2 and loc_model2 is not None:
        if errorbar:
            ax1.errorbar(phase_lc2, mag_lc2, err_lc2, fmt=marker2, markersize=1, errorevery=10)
            ax1.errorbar(phase_lc2-1, mag_lc2, err_lc2, fmt=marker2, markersize=1, errorevery=10, label='_nolegend_')
        else:
            ax1.plot(phase_lc2, mag_lc2, marker2, markersize=1)
            ax1.plot(phase_lc2 - 1, mag_lc2, marker2, markersize=1, label='_nolegend_')
        ax1.plot(phase_model2, mag_model2, line2, linewidth=2)
        ax1.plot(phase_model2 - 1, mag_model2, line2, linewidth=2, label='_nolegend_')

    if errorbar:
        ax2.errorbar(phase_lc1, mag_lc1, err_lc1, fmt=marker1, markersize=1, errorevery=10)
    else:
        ax2.plot(phase_lc1, mag_lc1, marker1, markersize=1)
    ax2.plot(phase_model1, mag_model1, line1, linewidth=2)
    ax2.set_xlim(phase_lim_secondary)
    if loc_lc2 and loc_model2 is not None:
        if errorbar:
            ax2.errorbar(phase_lc2, mag_lc2, err_lc2, fmt=marker2, markersize=1, errorevery=10)
        else:
            ax2.plot(phase_lc2, mag_lc2, marker2, markersize=1)
        ax2.plot(phase_model2, mag_model2, line2, linewidth=2)
        ax2.legend(['Light Curve 1', 'Model 1', 'Light Curve 2', 'Model 2'], markerscale=8)
    else:
        ax2.legend(['Light Curve', 'Model'], markerscale=8)
    if legend is not None:
        ax2.legend(legend, markerscale=8)

    if errorbar:
        ax3.errorbar(phase_lc1, o_c_1, err_lc1, fmt=marker1, markersize=1.5, errorevery=10)
        ax3.errorbar(phase_lc1-1, o_c_1, err_lc1, fmt=marker1, markersize=1.5, errorevery=10)
    else:
        ax3.plot(phase_lc1, o_c_1, marker1, markersize=1.5)
        ax3.plot(phase_lc1-1, o_c_1, marker1, markersize=1.5)
    ax3.plot([-0.5, 1.5], [0, 0], color='gray', linewidth=2)

    if plot_std == 'both':
        ax3.plot([-0.5, 1.5], [np.std(o_c_1), np.std(o_c_1)], '--', color=color1)
        ax3.plot([-0.5, 1.5], [-np.std(o_c_1), -np.std(o_c_1)], '--', color=color1)
        ax4.plot([-0.5, 1.5], [np.std(o_c_1), np.std(o_c_1)], '--', color=color1)
        ax4.plot([-0.5, 1.5], [-np.std(o_c_1), -np.std(o_c_1)], '--', color=color1)
        ax3.plot([-0.5, 1.5], [np.std(o_c_2), np.std(o_c_2)], '--', color=color2)
        ax3.plot([-0.5, 1.5], [-np.std(o_c_2), -np.std(o_c_2)], '--', color=color2)
        ax4.plot([-0.5, 1.5], [np.std(o_c_2), np.std(o_c_2)], '--', color=color2)
        ax4.plot([-0.5, 1.5], [-np.std(o_c_2), -np.std(o_c_2)], '--', color=color2)
    elif plot_std:
        ax3.plot([-0.5, 1.5], [np.std(o_c_1), np.std(o_c_1)], '--', color='gray')
        ax3.plot([-0.5, 1.5], [-np.std(o_c_1), -np.std(o_c_1)], '--', color='gray')
        ax4.plot([-0.5, 1.5], [np.std(o_c_1), np.std(o_c_1)], '--', color='gray')
        ax4.plot([-0.5, 1.5], [-np.std(o_c_1), -np.std(o_c_1)], '--', color='gray')
    if errorbar:
        ax4.errorbar(phase_lc1, o_c_1, err_lc1, fmt=marker1, markersize=1.5, errorevery=10)
    else:
        ax4.plot(phase_lc1, o_c_1, marker1, markersize=1.5)
    ax4.plot([-0.5, 1.5], [0, 0], color='gray', linewidth=2)
    if loc_lc2 and loc_model2 is not None:
        if errorbar:
            ax3.errorbar(phase_lc2, o_c_2, err_lc2, fmt=marker2, markersize=1.5, errorevery=10)
            ax3.errorbar(phase_lc2-1, o_c_2, err_lc2, fmt=marker2, markersize=1.5, errorevery=10)
            ax4.errorbar(phase_lc2, o_c_2, err_lc2, fmt=marker2, markersize=1.5, errorevery=10)
        else:
            ax3.plot(phase_lc2, o_c_2, marker2, markersize=1.5)
            ax3.plot(phase_lc2-1, o_c_2, marker2, markersize=1.5)
            ax4.plot(phase_lc2, o_c_2, marker2, markersize=1.5)

    ax1.set_ylabel('Relative Magnitude')
    ax3.set_xlabel('Phase')
    ax3.set_ylabel('O - C')
    ax4.set_xlabel('Phase')

    ax1.set_ylim(ax1.get_ylim()[::-1])
    if o_c_ylim is not None:
        ax3.set_ylim(o_c_ylim)
    else:
        ax3.set_ylim(ax3.get_ylim()[::-1])

    plt.show(block=False)

    plt.show()


def plot_many_lc(loc, period, phaselim1, phaselim2, subplot_text, figname):
    data = np.loadtxt(loc)
    time = data[:, 0]
    flux = data[:, 1]

    sort_idx = np.argsort(time)
    time = time[sort_idx]
    flux = flux[sort_idx]
    phase = np.mod(time, period) / period
    pdiff = np.diff(phase)

    cut_idx = np.where(pdiff < 0)[0]
    time_split = np.split(time, cut_idx)
    flux_split = np.split(flux, cut_idx)
    phase_split = np.split(phase, cut_idx)

    ncols = 2
    nrows = 9

    fig, axs = plt.subplots(nrows, ncols, figsize=(5, 10))
    k = 0
    for i in range(len(time_split)//2+3-nrows, len(time_split)//2+3):
        ax1, ax2 = axs[k, 0], axs[k, 1]

        timei = time_split[i]
        fluxi = flux_split[i]
        phasei = phase_split[i]

        ax1.plot(phasei, fluxi, 'r.', markersize=2)
        ax2.plot(phasei, fluxi, 'r.', markersize=2)

        ax1.text(0, 1, 'BJD '+"{:.2f}".format(timei[0])+subplot_text, transform=ax1.transAxes, verticalalignment='top')

        ax1.set_xlim(phaselim1)
        ax2.set_xlim(phaselim2)

        # y limits
        mask1 = (phasei > phaselim1[0]) & (phasei < phaselim1[1])
        mask2 = (phasei > phaselim2[0]) & (phasei < phaselim2[1])
        yvals1 = fluxi[mask1]
        yvals2 = fluxi[mask2]
        ax1.set_ylim([np.min(yvals1), np.max(yvals1)])
        ax2.set_ylim([np.min(yvals2), np.max(yvals2)])

        ylim1 = ax1.get_ylim()
        ylim_diff1 = ylim1[1] - ylim1[0]
        ylim2 = ax2.get_ylim()
        ylim_diff2 = ylim2[1] - ylim2[0]
        ax1.set_ylim([ylim1[0]-0.2*ylim_diff1, ylim1[1]+0.2*ylim_diff1])
        ax2.set_ylim([ylim2[0]-0.2*ylim_diff2, ylim2[1]+0.2*ylim_diff2])

        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        ax1.xaxis.set_ticks_position('none')
        ax2.xaxis.set_ticks_position('none')
        ax1.yaxis.set_ticks_position('none')
        ax2.yaxis.set_ticks_position('none')
        # ax1.axis("off")
        # ax2.axis("off")

        k += 1

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0)
    plt.subplots_adjust(hspace=0.0)
    plt.savefig(fname=figname+'.png', dpi=900)
    plt.savefig(fname=figname+'.pdf', dpi=300)
    plt.show()


# temperature_profile(5616, 5042, 0.727, 7.513)
# jktebop_model('jktebop_kepler/model.out', 'lcmag_kepler_full.txt', 'lcmag_kepler.txt', 'jktebop_kepler/rvA.dat',
#              'jktebop_kepler/rvB.dat', 63.32713, 54976.6351499878)
# jktebop_model(loc_model='tess/model.out', loc_lc='lcmag_tess.txt', loc_rvA='tess/rvA.dat',
#              loc_rvB='tess/rvB.dat', period=63.32713, initial_t=58712.9377353396)
# lc(loc='lcmag_tess_tot.txt', period=63.32713, model_loc='tess/model.out', phase_xlim=[-0.05, 0.4])
# psd(loc="datafiles/kasoc/8430105_psd.txt")

# lc_plot2('lcmag_kepler_reduced.txt', 'kepler_LTF/lc.KEPLER',
#         legend=['KASOC filtered Light Curve', 'LTF light curve'], ylim=[0.0225, -0.0025])

# obs_v_cal('kepler_kasfit/lc.KEPLER', 'kepler_kasfit/model.out')
if False:
    obs_v_cal_folded('JKTEBOP/kepler_LTF/lc.KEPLER', 'JKTEBOP/kepler_LTF/model.out', 'JKTEBOP/kepler_kasfit/lc.KEPLER',
                     'JKTEBOP/kepler_kasfit/model.out', legend=['JKTEBOP model LTF','JKTEBOP model KASOC',
                                                                'Kepler LTF LC', 'Kepler KASOC LC'],
                     o_c_ylim=[0.003, -0.003], marker1='y.', marker2='c.', line1='k--', line2='m-.', errorbar=True,
                     color1='k', color2='m', plot_std='both')
if False:
    obs_v_cal_folded('JKTEBOP/kepler_LTF/lc.KEPLER', 'JKTEBOP/kepler_LTF/model.out',
                     legend=['Kepler LTF LC', 'JKTEBOP model LTF'],
                     o_c_ylim=[0.003, -0.003], marker1='r.', marker2='c.', line1='k--', line2='m-.', errorbar=True)

if True:
    plot_many_lc('Data/unprocessed/kic8430105_kepler_unfiltered.txt', 63.32713, [0.1, 0.17], [0.445, 0.51], ' + 2400000',
                 '../figures/report/kepler/unfiltered_lc')
