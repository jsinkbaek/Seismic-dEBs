import numpy as np
import scipy.constants as c
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


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
    Calls planck_func and tess_spectral_response, and folds the two by summation.
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


Rb = 7.67996
# Ra = 1.12416
# Ra = 1.124480
Ra = 1.1245660185
Tb = 5042
Ta = 5700
print(luminosity_ratio(Rb, Ra, Tb, Ta))
print((Rb/Ra)**2 * (Tb/Ta)**4)

# Ta = find_T2(Rb, Ra, Tb, 29.75864)
# Ta = find_T2(Rb, Ra, Tb, 29.75335096)
Ta = find_T2(Rb, Ra, Tb, 29.7533890453)
print(Ta)
print(luminosity_ratio(Rb, Ra, Tb, Ta))
