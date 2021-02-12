import numpy as np
import scipy.constants as c
from scipy.interpolate import interp1d
import scipy.interpolate as interp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import subprocess


def planck_func(wl, T):
    """
    https://en.wikipedia.org/wiki/Planck%27s_law
    :param wl: numpy array containing wavelengths [Å]
    :param T: temperature [K]
    :return: spectral radiance [W/(sr m^2)]
    """
    wl = np.array(wl) * 1E-10       # Convert to array (if not) and from Ångström to metres
    spectral_radiance = (2*c.h*c.c**2 / wl**5) * 1/(np.exp(c.h*c.c / (wl*c.k*T)) - 1)
    return spectral_radiance


def tess_spectral_response(wl):
    """
    Loads normalized TESS spectral response values from file, interpolates and finds matching values for input
    wavelength.
    TESS spectral response values found at
    http://svo2.cab.inta-csic.es/theory/fps3/index.php?id=TESS/TESS.Red&&mode=browse&gname=TESS&gname2=TESS
    Key references:
    https://heasarc.gsfc.nasa.gov/docs/tess/the-tess-space-telescope.html
    Transiting Exoplanet Survey Satellite (TESS) (G. R. Ricker et al 2014)
        https://ui.adsabs.harvard.edu/abs/2014SPIE.9143E..20R/abstract

    :param wl: numpy array containing sampling wavelengths [Å]
    :return: interpolated spectral response values for the TESS bandpass
    """
    data = np.loadtxt("datafiles/TESS_TESS.Red.dat")
    intp = interp1d(data[:, 0], data[:, 1], kind='cubic')
    spectral_response = intp(wl)
    return spectral_response


def fold_summation(wl_a, wl_b, dw, T):
    """
    Calls planck_func and tess_spectral_response, and folds the two by summation of product.
    :param wl_a: initial wavelength [Å]
    :param wl_b: end wavelength [Å]
    :param dw: wavelength stepsize [Å]
    :param T: temperature [K]
    :return: np.sum(planck_func * tess_spectral_response) * dw
    """
    wl = np.linspace(wl_a, wl_b, int((wl_b-wl_a)//dw))
    dw_ = wl[1] - wl[0]     # actual dw used after integer division
    pfun = planck_func(wl, T)
    tsr = tess_spectral_response(wl)
    return np.sum(pfun*tsr)*dw_


def regular_summation(wl_a, wl_b, dw, T):
    """
    A simple version of fold_summation, assuming bolometric luminosity instead (ignoring TESS spectral response).
    To be used for testing purposes.
    :param wl_a: initial wavelength [Å]
    :param wl_b: end wavelength [Å]
    :param dw: wavelength stepsize [Å]
    :param T: temperature [K]
    :return: np.sum(planck_func) * dw
    """
    wl = np.linspace(wl_a, wl_b, int((wl_b - wl_a) // dw))
    dw_ = wl[1] - wl[0]  # actual dw used after integer division
    pfun = planck_func(wl, T)
    return np.sum(pfun)*dw_


def luminosity_ratio(R1, R2, T1, T2, wl_a=5670, wl_b=11270, dw=0.001):
    """
    Calculates luminosity ratio between two stars essentially using blackbody model L propto R²T⁴.
    Does this by calculating TESS integrated radiance given planck function and TESS spectral response.
    :param R1: radius of star 1 [same unit as R2]
    :param R2: radius of star 2 [same unit as R1]
    :param T1: effective temperature of star 1  [K]
    :param T2: effective temperature of star 2  [K]
    :param wl_a: initial wavelength to measure at [Å]
    :param wl_b: last wavelength to measure at [Å] (base them on TESS spectral response function)
    :param dw: wavelength stepsize. Defines precision of fold_summation [Å]
    :return: luminosity ratio L1/L2
    """
    integrated_radiance_1 = fold_summation(wl_a, wl_b, dw, T1)
    integrated_radiance_2 = fold_summation(wl_a, wl_b, dw, T2)
    return (R1/R2)**2 * (integrated_radiance_1/integrated_radiance_2)


def find_T2(R1, R2, T1, L_ratio):
    """
    Find T2, given R1, R2, T1, and L1/L2.
    :param R1: radius of star 1
    :param R2: radius of star 2 [same unit as R1]
    :param T1: effective temperature of star 1 [K]
    :param L_ratio: luminosity ratio L1/L2
    :return: effective temperature of star 2 [K]
    """
    def minimize_fun(T2):
        return np.abs(L_ratio - luminosity_ratio(R1, R2, T1, T2))
    optimize_result = minimize_scalar(minimize_fun, method='Bounded', bounds=(5000, 8000))
    if optimize_result.success:
        return optimize_result.x
    else:
        return


def read_jktebop_output(identifiers, loc='jktebop/param.out'):
    """
    Function to read final results from JKTEBOP output parameter file using string identifiers.
    Will only return the last occurrence of each value.
    :param identifiers: List of string identifiers for lines to return (e.g. [str(Eccentricity * cos(omega):)])
    :param loc:         location of output parameter file in project folder
    :return:            tuple with array of identifiers found matching, and array of their last value in file
    """
    name = np.array([])
    val = np.array([])
    used_id = np.array([])
    with open(loc) as f:
        for line in f:
            lsplit = line.strip().split("  ")
            if isinstance(lsplit, list):
                try:
                    for ident in identifiers:
                        print(lsplit[0])
                        if lsplit[0] == ident and ident not in used_id:
                            used_id = np.append(used_id, ident)
                            name = np.append(name, lsplit[0])
                            val = np.append(val, np.double(lsplit[-1]))
                        elif lsplit[0] == ident and ident in used_id:
                            replace_idx = np.argwhere(name == ident)
                            val[replace_idx] = np.double(lsplit[-1])
                except IndexError:
                    pass
    return name, val


def read_limb_darkening_parameters(logg_range=np.array([1, 5]), Trange=np.array([4000, 7000]), MH_range=-0.5,
                                   mTurb_range=2.0, loc='datafiles/tess_ldquad_table25.dat'):
    """
    Reads limb-darkening parameters from grid file for varying temperature and logg and outputs.
    :param logg_range:        log g range or requirement. Must be a numpy array for range, or float for requirement
    :param Trange:            temperature range or requirement
    :param MH_range:          metallicity range or requirement
    :param mTurb_range:       microturbulence range or requirement
    :param loc:               location of datafile in project folder
    :return:                  array of logg and temperature combinations, and their corresponding
                              limb-darkening parameters.
    """
    data = np.loadtxt(loc, usecols=(0, 1, 2, 3, 4, 5))
    logg = data[:, 0]
    Teff = data[:, 1]
    MH   = data[:, 2]
    mTurb= data[:, 3]
    if isinstance(mTurb_range, np.ndarray):
        mT_mask = (mTurb >= mTurb_range[0]) & (mTurb <= mTurb_range[1])
    else:
        mT_mask = mTurb == mTurb_range
        data = np.delete(data, 3, 1)        # delete column with singular data value
    if isinstance(MH_range, np.ndarray):
        MH_mask = (MH >= MH_range[0]) & (MH <= MH_range[1])
    else:
        MH_mask = MH == MH_range
        data = np.delete(data, 2, 1)
    if isinstance(Trange, np.ndarray):
        T_mask = (Teff >= Trange[0]) & (Teff <= Trange[1])
    else:
        T_mask = Teff == Trange
        data = np.delete(data, 1, 1)
    if isinstance(logg_range, np.ndarray):
        lg_mask = (logg >= logg_range[0]) & (logg <= logg_range[1])
    else:
        lg_mask = logg == logg_range
        data = np.delete(data, 0, 1)

    mask = lg_mask & T_mask & MH_mask & mT_mask
    data = data[mask, :]
    return data


def interpolated_LD_param(logg, Teff, MH, mTurb, logg_range=np.array([1, 5]), Trange=np.array([4000, 7000]),
                          MH_range=-0.5, mTurb_range=2.0, loc='datafiles/tess_ldquad_table25.dat'):
    """
    Interpolates limb-darkening parameters from grid data and evaluates in points given
    :param logg:        evaluation point
    :param Teff:        evaluation point
    :param MH:          evaluation point
    :param mTurb:       evaluation point
    :param logg_range:  range or requirement
    :param Trange:      range or requirement
    :param MH_range:    range or requirement
    :param mTurb_range: range or requirement
    :param loc:         datafile location in project folder
    :return:            limbdarkening parameters a and b (quadratic LS fit values for default file)
    """
    data = read_limb_darkening_parameters(logg_range, Trange, MH_range, mTurb_range, loc)
    vals = data[:, -2:]
    points = data[:, 0:-2]
    eval_points = np.array([])
    if isinstance(logg_range, np.ndarray):
        eval_points = np.append(eval_points, logg)
    if isinstance(Trange, np.ndarray):
        eval_points = np.append(eval_points, Teff)
    if isinstance(MH_range, np.ndarray):
        eval_points = np.append(eval_points, MH)
    if isinstance(mTurb_range, np.ndarray):
        eval_points = np.append(eval_points, mTurb)

    eval_points = np.reshape(eval_points, (1, eval_points.size))
    res = interp.griddata(points, vals, eval_points, method='cubic')

    return res


def jktebop_iterator():
    subprocess.run('cd jktebop && make clean && make')
    _, jktebop_vals = read_jktebop_output(['Log surface gravity of star A (cgs):',
                                           'Log surface gravity of star B (cgs):', 'Radius of star A (Rsun)',
                                           'Radius of star B (Rsun)', 'Stellar light ratio (phase 0.1706):'])
    [loggMS, loggRG, R_MS, R_RG, L_ratio] = jktebop_vals


print(interpolated_LD_param(4.23, 5620, -0.5, 2.0))

Rb = 7.5436984091
# Ra = 1.12416
# Ra = 1.124480
Ra = 1.1318818664
Tb = 5042

# Ta = find_T2(Rb, Ra, Tb, 29.75864)
# Ta = find_T2(Rb, Ra, Tb, 29.75335096)
# Ta = find_T2(Rb, Ra, Tb, 29.7533595146)
# print(Ta)
# print(luminosity_ratio(Rb, Ra, Tb, Ta))


