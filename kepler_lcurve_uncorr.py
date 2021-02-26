import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy.polynomial import Polynomial


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


def plt_inspect(x, y, fmt='r.'):
    plt.figure()
    plt.plot(x, y, fmt, markersize=0.7)
    coords = plt.ginput(n=2, timeout=0, show_clicks=True, mouse_add=1, mouse_stop=3, mouse_pop=2)
    return coords


def inspection_loop(ph, fl, tm, fit_mask, res_mask, polydeg):
    (phn, fln, tmn), p_ = poly_trendfit(ph, fl, tm, fit_mask, res_msk=res_mask, deg=polydeg)
    while True:
        ins_res = plt_inspect(phn, fln)
        if ins_res:
            ins_mask = (ph > ins_res[0][0]) & (ph < ins_res[1][0])
            fit_mask = ins_mask & fit_mask
            res_mask = ins_mask & res_mask
        else:
            inpt = input("Input b to break loop. Input d to delete. To change polynomial degree, input number (ex. 2).")
            if inpt == 'b':
                break
            elif inpt == 'd':
                return ()
            else:
                try:
                    polydeg = int(inpt)
                except ValueError:
                    print('Input was neither b or an integer number.')
        (phn, fln, tmn), p_ = poly_trendfit(ph, fl, tm, fit_mask, res_msk=res_mask, deg=polydeg)
    plt.close('all')
    return (phn, fln, tmn), p_


# # # Start of Script # # #
# Load data
with fits.open("datafiles/kasoc/kplr008430105_kasoc-ts_llc_v1.fits") as hdul:
    print(hdul.info())
    hdu = hdul[1]
    print(hdu.columns)
    time = hdu.data['TIME']
    flux_seism = hdu.data['FLUX']
    err_seism = hdu.data['FLUX_ERR']
    corr_long = hdu.data['XLONG']
    corr_transit = hdu.data['XPHASE']
    corr_short = hdu.data['XSHORT']
    corr_pos = hdu.data['XPOS']
    corr_full = hdu.data['FILTER']
    kasoc_qflag = hdu.data['KASOC_FLAG']

# Remove nan's
nan_mask = np.isnan(time) | np.isnan(flux_seism) | np.isnan(corr_full)
time = time[~nan_mask]
flux_seism = flux_seism[~nan_mask]
err_seism = err_seism[~nan_mask]
corr_long = corr_long[~nan_mask]
corr_transit = corr_transit[~nan_mask]
corr_short = corr_short[~nan_mask]
corr_full = corr_full[~nan_mask]
corr_pos = corr_pos[~nan_mask]


# Make uncorrected relative flux
flux = (flux_seism * 1E-6 + 1) * corr_full
# Correct flux for transit
flux_transit = flux / (corr_long + corr_short)
# Do phase fold
period = 63.32713
phase = np.mod(time, period) / period

# # # Cut uncorrected light curve down to eclipses, normalize and plot # # #
# Split dataset into separate orbits
phasecut_mask = np.diff(phase) < -0.8
phasecut_idxs = np.where(phasecut_mask)[0]
flux_split = np.split(flux, phasecut_idxs)
phase_split = np.split(phase, phasecut_idxs)
time_split = np.split(time, phasecut_idxs)


# Correct and normalize flux (either with eclipse normalization for quick plot, or polynomial local trend fitting)
norm_flux1 = np.array([])
norm_flux2 = np.array([])
norm_phase = np.array([])
flux_p1 = np.array([])
flux_p2 = np.array([])
phase_p1 = np.array([])
phase_p2 = np.array([])
time_p1 = np.array([])
time_p2 = np.array([])
p1 = []
p2 = []

for i in range(0, len(phase_split)):
    current_phases = phase_split[i]
    current_flux = flux_split[i]
    current_time = time_split[i]

    eclipse1_mask = (current_phases > 0.1269) & (current_phases < 0.1485)
    norm_flux1_ = current_flux / np.median(current_flux[eclipse1_mask])
    norm_flux1 = np.append(norm_flux1, norm_flux1_)

    eclipse2_mask = (current_phases > 0.4724) & (current_phases < 0.4851)
    norm_flux2_ = current_flux / np.median(current_flux[eclipse2_mask])
    norm_flux2 = np.append(norm_flux2, norm_flux2_)

    norm_phase = np.append(norm_phase, current_phases)

    # fit_mask1 = ((current_phases > 0.04) & (current_phases < 0.12292)) \
    #             | ((current_phases > 0.15164) & (current_phases < 0.24))
    fit_mask1 = ((current_phases > 0.02) & (current_phases < 0.12292)) \
                | ((current_phases > 0.15164) & (current_phases < 0.3))
    # fit_mask2 = ((current_phases > 0.38) & (current_phases < 0.46269)) \
    #             | ((current_phases > 0.49391) & (current_phases < 0.58))
    fit_mask2 = ((current_phases > 0.34) & (current_phases < 0.46269)) \
                | ((current_phases > 0.49391) & (current_phases < 0.65))
    res_mask1 = (current_phases > 0.04) & (current_phases < 0.24)
    res_mask2 = (current_phases > 0.38) & (current_phases < 0.58)

    if norm_flux1_[~np.isnan(norm_flux1_)].size > 30:
        deg = 2
        inspect_res = inspection_loop(current_phases, current_flux, current_time, fit_mask1,
                                                  res_mask=res_mask1, polydeg=deg)
        if inspect_res:
            (phn1, fln1, tmn1), p1_ = inspect_res
            phase_p1, flux_p1, time_p1 = np.append(phase_p1, phn1), np.append(flux_p1, fln1), np.append(time_p1, tmn1)
            p1.append(p1_)

    if norm_flux2_[~np.isnan(norm_flux2_)].size > 30:
        deg = 2
        inspect_res = inspection_loop(current_phases, current_flux, current_time, fit_mask2,
                                                  res_mask=res_mask2, polydeg=deg)
        if inspect_res:
            (phn2, fln2, tmn2), p2_ = inspect_res
            phase_p2, flux_p2, time_p2 = np.append(phase_p2, phn2), np.append(flux_p2, fln2), np.append(time_p2, tmn2)
            p2.append(p2_)


t0 = 54976
print(norm_flux1[np.isnan(norm_flux1)].shape)
print(norm_flux2[np.isnan(norm_flux2)].shape)

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
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(phase_p1, flux_p1, 'r.', markersize=0.7)
    ax2.plot(phase_p2, flux_p2, 'r.', markersize=0.7)
    plt.show()

if True:
    plt.figure()
    plt.plot(time_p1, flux_p1, 'r.', markersize=0.7)
    plt.show(block=False)
    plt.figure()
    plt.plot(time_p2, flux_p2, 'b.', markersize=0.7)
    plt.show()