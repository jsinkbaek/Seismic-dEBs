import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.io import fits
from numpy.polynomial import Polynomial
from scipy.signal import medfilt
import os

# matplotlib.use('Qt5Agg')
print(matplotlib.get_backend())


# # # Functions for script # # #
def poly_trendfit(ph, fl, tm, msk, deg=2, res_msk=None):
    """
    :param ph: phase
    :param fl: flux
    :param tm: time
    :param msk: mask of data to include
    :param deg: polynomial fit degree
    :param res_msk: mask that limits the extend of the result
    :return:
    """
    ph_cut, fl_cut, tm_cut = ph[msk], fl[msk], tm[msk]
    pfit = Polynomial.fit(tm_cut, fl_cut, deg=deg)
    fl_res = fl / pfit(tm)
    tm_res, ph_res = tm, ph
    if res_msk is not None:
        ph_res, fl_res, tm_res = ph_res[res_msk], fl_res[res_msk], tm_res[res_msk]
    return (ph_res, fl_res, tm_res), pfit


def poly_plt(fl, tm, fln, tmn, pfit, tm0, tm1, fignr, component, res_mask, fit_mask, include_kasoc=False):
    matplotlib.rcParams.update({'font.size': 17})
    tmfit = np.linspace(tm[0], tm[-1], 1000)
    pvals = pfit(tmfit)
    # Median filter rejection for plot limits
    mfilt = medfilt(fl, 5)
    lim_mask = (np.abs(fl - mfilt) < np.std(fl - mfilt)) & ((tm > tm0) & (tm < tm1))
    ylow = np.min(fl[lim_mask])
    yhigh = np.max(fl[lim_mask])
    yavg = np.average(fl[lim_mask])

    fig = plt.figure(figsize=[6.4*1.5, 4.8*1.5])
    plt.ticklabel_format(axis='x', style='plain', useOffset=False)
    plt.plot(tm, fl, 'r.', markersize=1.0)
    plt.plot(tm[fit_mask], fl[fit_mask]-yavg*0.002, 'b.', markersize=1.0)
    plt.plot(tm[res_mask], fl[res_mask]-yavg*0.004, 'g.', markersize=1.0)
    plt.plot(tmfit, pvals, 'k--')
    plt.xlim([tm0, tm1])
    plt.ylim([ylow-ylow*0.008, yhigh +0.008*yhigh])
    plt.xlabel('Time: BJD - 2400000')
    plt.ylabel('Flux [e-/s]')
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.legend(['Uncorrected light curve', 'Data used for fit', 'Data for corrected LC', 'Chosen LTF polynomial'],
                markerscale=8)
    plt.savefig(fname='../../figures/LTF_pdcsap/8430105/fig_'+str(fignr)+component+'1', orientation='landscape', dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=[6.4*1.5, 4.8*1.5])
    plt.plot(tmn, fln, 'r.', markersize=2.5)
    plt.xlim([np.min(tmn)-0.5, np.max(tmn)+0.5])
    plt.xlabel('Time: BJD - 2400000')
    plt.ylim([0.97, 1.005])
    plt.ylabel('Relative Flux')
    if include_kasoc:
        tm_kas, fl_kas, fl_err_kas = load_kasoc_lc()
        plt.plot(tm_kas, fl_kas, 'b.', markersize=2)
        plt.legend(['LTF corrected light curve', 'KASOC filtered light curve'], markerscale=8)
    else:
        plt.legend(['LTF corrected light curve'], markerscale=8)
    plt.ticklabel_format(axis='x', style='plain', useOffset=False)
    plt.savefig(fname='../../figures/LTF_pdcsap/8430105/fig_'+str(fignr)+component+'2', orientation='landscape', dpi=150)
    plt.close(fig)


def plt_inspect(ph, fl, tm, phn, fln, fit_mask, res_mask, pfit, tm0, tm1):
    plt.figure()
    plt.plot(phn, fln, 'r.', markersize=1.0)
    plt.plot(phn, np.ones(shape=phn.shape), 'k--')
    plt.show(block=False)

    matplotlib.rcParams.update({'font.size': 17})
    tmfit = np.linspace(tm[0], tm[-1], 1000)
    phfit = np.linspace(np.min(ph), np.max(ph), 1000)
    pvals = pfit(tmfit)
    # Median filter rejection for plot limits
    mfilt = medfilt(fl, 5)
    ph_lims = ph[(tm > tm0) & (tm < tm1)]
    lim_mask = (np.abs(fl - mfilt) < np.std(fl - mfilt)) & ((tm > tm0) & (tm < tm1))
    ylow = np.min(fl[lim_mask])
    yhigh = np.max(fl[lim_mask])
    yavg = np.average(fl[lim_mask])

    fig = plt.figure(figsize=[6.4*1.5, 4.8*1.5])
    plt.ticklabel_format(axis='x', style='plain', useOffset=False)
    plt.plot(ph, fl, 'r.', markersize=1.0)
    plt.plot(ph[fit_mask], fl[fit_mask] - yavg * 0.002, 'b.', markersize=1.0)
    plt.plot(ph[res_mask], fl[res_mask] - yavg * 0.004, 'g.', markersize=1.0)
    plt.plot(phfit, pvals, 'k--')
    plt.xlim([ph_lims[0], ph_lims[-1]])
    plt.ylim([ylow - ylow * 0.005, yhigh + 0.005 * yhigh])
    plt.xlabel('Phase')
    plt.ylabel('Flux [e-/s]')
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.legend(['Uncorrected light curve', 'Data used for fit', 'Data for corrected LC', 'Chosen LTF polynomial'],
               markerscale=8)
    coords = plt.ginput(n=2, timeout=0, show_clicks=True, mouse_add=1, mouse_stop=3, mouse_pop=2)
    return coords


def inspection_loop(ph, fl, tm, fit_mask, res_mask, deg, tm0, tm1):
    (phn, fln, tmn), pfit = poly_trendfit(ph, fl, tm, fit_mask, res_msk=res_mask, deg=deg)
    fit_mask_ = fit_mask
    res_mask_ = res_mask
    while True:
        ins_res = plt_inspect(ph, fl, tm, phn, fln, fit_mask_, res_mask_, pfit, tm0, tm1)
        if ins_res:

            ins_mask = (ph > ins_res[0][0]) & (ph < ins_res[1][0])
            fit_mask_ = ins_mask & fit_mask
            res_mask_ = ins_mask & res_mask
        else:
            inpt = input("Input b to break loop. Input d to delete. To change polynomial degree, input number (ex. 2).")
            if inpt == 'b':
                break
            elif inpt == 'd':
                return ()
            else:
                try:
                    deg = int(inpt)
                except ValueError:
                    print('Input was neither b or an integer number.')
        (phn, fln, tmn), pfit = poly_trendfit(ph, fl, tm, fit_mask_, res_msk=res_mask_, deg=deg)
    plt.close('all')
    return (phn, fln, tmn), pfit, fit_mask_, res_mask_


def save_phase(ph, fl, tm, i_, pfit, component, loc='../Data/processed/KIC8430105/pdcsap_trendfit_results/'):
    """

    :param ph: phase
    :param fl: flux
    :param tm: time
    :param i_: integer designation of phase to save
    :param pfit: polynomial fit results
    :param component: string 'A' or 'B' designating first (full) or second eclipse
    :param loc: datafile folder location
    """
    data_ = np.empty((ph.size, 3))
    data_[:, 0], data_[:, 1], data_[:, 2] = ph, tm, fl
    np.savetxt(loc+str(i_)+component+'.dat', data_)
    np.savetxt(loc+str(i_)+component+'.pfit', pfit.coef)


def load_phase(i_, component, loc='../Data/processed/KIC8430105/pdcsap_trendfit_results/'):
    """
    :param i_: integer designating phase to load
    :param component: string 'A' or 'B' designating first (full) or second eclipse
    :param loc: datafile folder location
    """
    try:
        data_ = np.loadtxt(loc+str(i_)+component+'.dat')
        ph, tm, fl = data_[:, 0], data_[:, 1], data_[:, 2]
        return ph, tm, fl
    except OSError as e:
        if str(e) != loc+str(i_)+component+".dat"+" not found.":
            raise e
        else:
            print('Datafile '+str(i_)+component+' is missing. Was probably skipped intentionally.')


def load_kasoc_lc(loc='../Data/processed/KIC8430105/lcflux_kasoc_reduced_full.txt'):
    """
    Loads a previously corrected lightkurve using parts of KASOC filter (see kepler_lcurve_corr.py).
    """
    data = np.loadtxt(loc)
    tm, flx, flx_err = data[:, 0], data[:, 1], data[:, 2]
    return tm, flx, flx_err


# # # Start of Script # # #
# Load data (Either previous trend fitted dataset, or KASOC datafile)
load_previous = True
if load_previous:
    flux_p1 = np.array([])
    flux_p2 = np.array([])
    flux_p12 = np.array([])
    phase_p1 = np.array([])
    phase_p2 = np.array([])
    phase_p12 = np.array([])
    time_p1 = np.array([])
    time_p2 = np.array([])
    time_p12 = np.array([])
    for i in range(1, 24):
        load1 = load_phase(i, 'A')
        if load1 is not None:
            phase1, time1, flux1 = load1
            phase_p1, time_p1, flux_p1 = np.append(phase_p1, phase1), np.append(time_p1, time1), np.append(flux_p1,
                                                                                                           flux1)
            phase_p12 = np.append(phase_p12, phase1)
            flux_p12 = np.append(flux_p12, flux1)
            time_p12 = np.append(time_p12, time1)
        load2 = load_phase(i, 'B')
        if load2 is not None:
            phase2, time2, flux2 = load2
            phase_p2, time_p2, flux_p2 = np.append(phase_p2, phase2), np.append(time_p2, time2), np.append(flux_p2,
                                                                                                           flux2)
            flux_p12 = np.append(flux_p12, flux2)
            phase_p12 = np.append(phase_p12, phase2)
            time_p12 = np.append(time_p12, time2)
else:
    pdcsap = np.array([])
    time = np.array([])
    pdcsap_err = np.array([])
    for fname in os.listdir('../Data/unprocessed/mast/kepler-kic8430105/'):
        with fits.open('../Data/unprocessed/mast/kepler-kic8430105/'+fname) as hdul:
            hdu = hdul[1]
            time_temp = hdu.data['TIME']
            pdcsap_temp = hdu.data['PDCSAP_FLUX']
            pdcsap_err_temp = hdu.data['PDCSAP_FLUX_ERR']
            nan_mask = np.isnan(time_temp) | np.isnan(pdcsap_temp) | np.isnan(pdcsap_err_temp)
            time_temp = time_temp[~nan_mask]
            pdcsap_temp = pdcsap_temp[~nan_mask]
            pdcsap_err_temp = pdcsap_err_temp[~nan_mask]
            pdcsap = np.append(pdcsap, pdcsap_temp)
            time = np.append(time, time_temp)
            pdcsap_err = np.append(pdcsap_err, pdcsap_err_temp)

    sort_idx = np.argsort(time)
    time = time[sort_idx]
    pdcsap = pdcsap[sort_idx]
    pdcsap_err = pdcsap_err[sort_idx]

    # Do phase fold
    period = 63.32713
    phase = np.mod(time, period) / period

    # # # Cut uncorrected light curve down to eclipses and do Local Trend Fitting # # #
    # Split dataset into separate orbits
    phasecut_mask = np.diff(phase) < -0.8
    phasecut_idxs = np.where(phasecut_mask)[0]
    flux_split = np.split(pdcsap, phasecut_idxs)
    phase_split = np.split(phase, phasecut_idxs)
    time_split = np.split(time, phasecut_idxs)
    for i in range(0, len(phase_split)):
        plt.plot(phase_split[i], flux_split[i], '.')
    plt.show()

    # Correct and normalize flux (either with eclipse normalization for quick plot, or polynomial local trend fitting)
    norm_flux1 = np.array([])
    norm_flux2 = np.array([])
    norm_phase = np.array([])
    flux_p1 = np.array([])
    flux_p2 = np.array([])
    flux_p12 = np.array([])
    phase_p1 = np.array([])
    phase_p2 = np.array([])
    phase_p12 = np.array([])
    time_p1 = np.array([])
    time_p2 = np.array([])
    time_p12 = np.array([])
    p1 = []
    p2 = []

    for i in range(0, len(phase_split)):
        current_phases = phase_split[i]
        current_flux = flux_split[i]
        current_time = time_split[i]

        diff1 = 0.12292-0.02
        diff2 = 0.3-0.15164
        diff3 = 0.46269-0.34
        diff4 = 0.65-0.49391
        fit_mask1 = ((current_phases > 0.253-diff1) & (current_phases < 0.253)) \
                    | ((current_phases > 0.285) & (current_phases < 0.285+diff2))
        fit_mask2 = ((current_phases > 0.592-diff3) & (current_phases < 0.592)) \
                    | ((current_phases > 0.629) & (current_phases < 0.629+diff4))
        res_mask1 = (current_phases > 0.2) & (current_phases < 0.35)
        res_mask2 = (current_phases > 0.53) & (current_phases < 0.7)

        # # Polynomial local trend fitting normalization # #
        if current_phases[fit_mask1].size > 10:
            polydeg = 2
            ct_masked = current_time[fit_mask1]
            inspect_res = inspection_loop(current_phases, current_flux, current_time, fit_mask1,
                                          res_mask=res_mask1, deg=polydeg, tm0=ct_masked[0], tm1=ct_masked[-1])
            if inspect_res:
                (phn1, fln1, tmn1), p1_, fmsk1, rmsk1 = inspect_res
                phase_p1, flux_p1, time_p1 = np.append(phase_p1, phn1), np.append(flux_p1, fln1), np.append(time_p1,
                                                                                                            tmn1)
                p1.append(p1_)
                flux_p12 = np.append(flux_p12, fln1)
                phase_p12 = np.append(phase_p12, phn1)
                time_p12 = np.append(time_p12, tmn1)
                save_phase(phn1, fln1, tmn1, i, pfit=p1_, component='A')

                poly_plt(current_flux, current_time, fln1, tmn1, p1_, ct_masked[0], ct_masked[-1], i, 'A',
                         rmsk1, fmsk1, include_kasoc=True)
        if current_phases[fit_mask2].size > 10:
            polydeg = 2
            ct_masked = current_time[fit_mask2]
            inspect_res = inspection_loop(current_phases, current_flux, current_time, fit_mask2,
                                          res_mask=res_mask2, deg=polydeg, tm0=ct_masked[0], tm1=ct_masked[-1])
            if inspect_res:
                (phn2, fln2, tmn2), p2_, fmsk2, rmsk2 = inspect_res
                phase_p2, flux_p2, time_p2 = np.append(phase_p2, phn2), np.append(flux_p2, fln2), np.append(time_p2,
                                                                                                            tmn2)
                p2.append(p2_)
                flux_p12 = np.append(flux_p12, fln2)
                phase_p12 = np.append(phase_p12, phn2)
                time_p12 = np.append(time_p12, tmn2)
                save_phase(phn2, fln2, tmn2, i, pfit=p2_, component='B')
                poly_plt(current_flux, current_time, fln2, tmn2, p2_, ct_masked[0], ct_masked[-1], i, 'B', rmsk2,
                         fmsk2, include_kasoc=True)

t0 = 54976

if False:
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(norm_phase, norm_flux1, 'r.', markersize=0.7)
    ax1.set_xlim([0.04, 0.24])
    ax2.plot(norm_phase, norm_flux2, 'r.', markersize=0.7)
    ax2.set_xlim([0.38, 0.58])
    plt.show()

if False:
    plt.figure()
    plt.plot(norm_phase, norm_flux1, 'r.', markersize=0.7)
    plt.plot(norm_phase, norm_flux2, 'b.', markersize=0.7)
    plt.show()

if True:
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(phase_p1, flux_p1, 'r.', markersize=0.7)
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Relative Flux')
    ax2.plot(phase_p2, flux_p2, 'r.', markersize=0.7)
    ax2.set_xlabel('Phase')
    plt.show()

if False:
    plt.figure()
    plt.plot(time_p1, flux_p1, 'r.', markersize=0.7)
    plt.show(block=False)
    plt.figure()
    plt.plot(time_p2, flux_p2, 'b.', markersize=0.7)
    plt.show()

# Measure spread within full eclipse as estimator of flux error
rmse_measure_mask = (phase_p1 > 0.2569) & (phase_p1 < 0.2798)
rmse_used_vals = flux_p1[rmse_measure_mask]
mean_val = np.mean(rmse_used_vals)
error = np.sqrt(np.sum((mean_val - rmse_used_vals)**2) / rmse_used_vals.size)

if True:
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.errorbar(phase_p1, flux_p1, error, fmt='k.', ecolor='gray', markersize=0.5, elinewidth=0.1, errorevery=10)
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Relative Flux')
    ax2.errorbar(phase_p2, flux_p2, error, fmt='k.', ecolor='gray', markersize=0.5, elinewidth=0.1, errorevery=10)
    ax2.set_xlabel('Phase')
    plt.show()


# # # Add fictitious contamination # # #
flux_p1 = flux_p1 * 0.9 + 0.1
flux_p2 = flux_p2 * 0.9 + 0.1
flux_p12 = flux_p12 * 0.9 + 0.1

# # # Convert to magnitudes # # #
m_1     = -2.5*np.log10(flux_p1)
m_2     = -2.5*np.log10(flux_p2)
m       = -2.5*np.log10(flux_p12)
m_err_1 = np.abs(-2.5/np.log(10) * (error/flux_p1))
m_err_2 = np.abs(-2.5/np.log(10) * (error/flux_p2))
m_err   = np.abs(-2.5/np.log(10) * (error/flux_p12))
print(m.shape)

if True:
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.errorbar(phase_p1, m_1, m_err_1, fmt='k.', ecolor='gray', markersize=0.5, elinewidth=0.1, errorevery=10)
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Relative Magnitude')
    ax1.set_ylim([0.020, -0.003])
    ax2.errorbar(phase_p2, m_2, m_err_2, fmt='k.', ecolor='gray', markersize=0.5, elinewidth=0.1, errorevery=10)
    ax2.set_xlabel('Phase')
    plt.show()

if True:
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.scatter(phase_p12, m, color='gray', marker='.', s=0.5)
    ax1.set_xlabel('Phase')
    ax1.set_ylim([0.020, -0.003])
    ax1.set_xlim([np.min(phase_p1), np.max(phase_p1)])
    ax1.set_ylabel('Relative Magnitude')
    ax2.scatter(phase_p12, m, color='gray', marker='.', s=0.5)
    ax2.set_xlabel('Phase')
    ax2.set_ylim([0.020, -0.003])
    ax2.set_xlim([np.min(phase_p2), np.max(phase_p2)])

mask = ~np.isnan(m) & ~np.isnan(m_err) & ~np.isnan(time_p12) & ~np.isnan(phase_p12)
m = m[mask]
flux_p12 = flux_p12[mask]
time_p12 = time_p12[mask]
phase_p12 = phase_p12[mask]
m_err = m_err[mask]
print(m.shape)

# Cut fit data down to bone
# diff_x1 = 0.156 - 0.117
# diff_x2 = 0.500 - 0.457
diff_x1 = 0.025
diff_x2 = 0.025
mask = ((phase_p12 > 0.268-diff_x1) & (phase_p12 < 0.268+diff_x1)) | \
       ((phase_p12 > 0.6087-diff_x2) & (phase_p12 < 0.6087+diff_x2))
m = m[mask]
flux_p12 = flux_p12[mask]
time_p12 = time_p12[mask]
phase_p12 = phase_p12[mask]
m_err = m_err[mask]
print(m.shape)

if True:
    ax1.scatter(phase_p12, m, marker='.', color='k', s=0.5)
    ax2.scatter(phase_p12, m, marker='.', color='k', s=0.5)
    plt.show()

if True:
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.errorbar(phase_p12, m, m_err, fmt='k.', ecolor='gray', markersize=0.5, elinewidth=0.1, errorevery=10)
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Relative Magnitude')
    ax1.set_xlim([0.268-diff_x1-0.0002, 0.268+diff_x1+0.0002])
    ax1.set_ylim([0.020, -0.003])
    ax2.errorbar(phase_p12, m, m_err, fmt='k.', ecolor='gray', markersize=0.5, elinewidth=0.1, errorevery=10)
    ax2.set_xlabel('Phase')
    ax2.set_xlim([0.6087-diff_x2-0.0002, 0.6087+diff_x2+0.0002])
    plt.show()

save_data = np.zeros((m.size, 3))
print(save_data.shape)
save_data[:, 0] = time_p12 + 54833.0
save_data[:, 1] = m
save_data[:, 2] = m_err
np.savetxt('../Data/processed/KIC8430105/lcmag_kepler_pdcsap_3lbi.txt', save_data)
save_data[:, 1] = flux_p12
save_data[:, 2] = np.ones(flux_p12.shape) * error
np.savetxt('../Data/processed/KIC8430105/lcflux_kepler_pdcsap_3lbi.txt', save_data)

