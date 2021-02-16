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


def kepler_spectral_response(wl):
    data = np.loadtxt("datafiles/Kepler_Kepler.K.dat")
    intp = interp1d(data[:, 0], data[:, 1], kind='cubic')
    spectral_response = intp(wl)
    return spectral_response


def spectral_response_limits(spectral_response):
    if spectral_response==tess_spectral_response:
        data = np.loadtxt("datafiles/TESS_TESS.Red.dat")
    elif spectral_response==kepler_spectral_response:
        data = np.loadtxt("datafiles/Kepler_Kepler.K.dat")
    else:
        raise AttributeError("Unknown spectral_response")
    wl_a, wl_b = data[0, 0], data[-1, 0]
    return wl_a, wl_b


def fold_summation(dw, T, spectral_response=tess_spectral_response):
    """
    Calls planck_func and tess_spectral_response, and folds the two by summation of product.
    :param dw:                  wavelength stepsize [Å]
    :param T:                   temperature [K]
    :param spectral_response:   spectral response function (e.g. tess_spectral_response)
    :return: np.sum(planck_func * tess_spectral_response) * dw
    """
    wl_a, wl_b = spectral_response_limits(spectral_response)
    wl = np.linspace(wl_a, wl_b, int((wl_b-wl_a)//dw))
    dw_ = wl[1] - wl[0]     # actual dw used after integer division
    pfun = planck_func(wl, T)
    sr = spectral_response(wl)
    return np.sum(pfun*sr)*dw_


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


def luminosity_ratio(R1, R2, T1, T2, spectral_response, dw=0.001):
    """
    Calculates luminosity ratio between two stars essentially using blackbody model L propto R²T⁴.
    Does this by calculating TESS integrated radiance given planck function and TESS spectral response.
    :param R1: radius of star 1 [same unit as R2]
    :param R2: radius of star 2 [same unit as R1]
    :param T1: effective temperature of star 1  [K]
    :param T2: effective temperature of star 2  [K]
    :param spectral_response: spectral response function of detector
    :param dw: wavelength stepsize. Defines precision of fold_summation [Å]
    :return: luminosity ratio L1/L2
    """
    integrated_radiance_1 = fold_summation(dw, T1, spectral_response)
    integrated_radiance_2 = fold_summation(dw, T2, spectral_response)
    return (R1/R2)**2 * (integrated_radiance_1/integrated_radiance_2)


def find_T2(R1, R2, T1, L_ratio, spectral_response):
    """
    Find T2, given R1, R2, T1, and L1/L2.
    :param R1:                  radius of star 1
    :param R2:                  radius of star 2 [same unit as R1]
    :param T1:                  effective temperature of star 1 [K]
    :param L_ratio:             luminosity ratio L1/L2
    :param spectral_response:   spectral response function to use for satellite detector
    :return: effective temperature of star 2 [K]
    """
    def minimize_fun(T2):
        return np.abs(L_ratio - luminosity_ratio(R1, R2, T1, T2, spectral_response=spectral_response))
    optimize_result = minimize_scalar(minimize_fun, method='Bounded', bounds=(4000, 8000))
    if optimize_result.success:
        return optimize_result.x
    else:
        return optimize_result.message


def read_jktebop_output(identifiers, loc='jktebop_tess/param.out'):
    """
    Convenience function to read results from JKTEBOP output parameter file using string identifiers.
    Will only return the last occurrence of each value in the file.
    :param identifiers: List of string identifiers for lines to return (e.g. [str(Eccentricity * cos(omega):)])
    :param loc:         location of output parameter file in project folder
    :return:            tuple with array of identifiers found matching, and array of their last value in file
    """
    name = np.array([])
    val = np.array([])
    used_id = np.array([])
    with open(loc, "r") as f:
        for line in f:
            lsplit = line.strip().split("  ")
            if isinstance(lsplit, list):
                try:
                    for ident in identifiers:
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


def read_limb_darkening_parameters(logg_range=np.array([1, 5]), Trange=np.array([2000, 9000]), MH_range=-0.5,
                                   mTurb_range=2.0, loc='datafiles/tess_ldquad_table25.dat'):
    """
    Convenience function. Reads limb-darkening parameters from grid file for varying temperature and logg and outputs.
    :param logg_range:        log g range or requirement. Must be a numpy array for range, or float for requirement
    :param Trange:            temperature range or requirement
    :param MH_range:          metallicity range or requirement
    :param mTurb_range:       microturbulence range or requirement
    :param loc:               location of datafile in project folder
    :return:                  array of logg and temperature combinations, and their corresponding
                              limb-darkening parameters.
    """
    if loc == 'datafiles/tess_ldquad_table25.dat':
        data = np.loadtxt(loc, usecols=(0, 1, 2, 3, 4, 5))
        logg = data[:, 0]
        Teff = data[:, 1]
        MH   = data[:, 2]
        mTurb= data[:, 3]
        if isinstance(mTurb_range, np.ndarray):
            mT_mask = (mTurb >= mTurb_range[0]) & (mTurb <= mTurb_range[1])
        else:
            mT_mask = mTurb == mTurb_range
            data = np.delete(data, 3, 1)  # delete column with singular data value
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

    elif loc == 'datafiles/kepler_sing_table.dat':
        data = np.loadtxt(loc, usecols=(0, 1, 2, 4, 5))
        Teff = data[:, 0]
        logg = data[:, 1]
        MH = data[:, 2]
        if isinstance(MH_range, np.ndarray):
            MH_mask = (MH >= MH_range[0]) & (MH <= MH_range[1])
        else:
            MH_mask = MH == MH_range
            data = np.delete(data, 2, 1)
        if isinstance(logg_range, np.ndarray):
            lg_mask = (logg >= logg_range[0]) & (logg <= logg_range[1])
        else:
            lg_mask = logg == logg_range
            data = np.delete(data, 1, 1)
        if isinstance(Trange, np.ndarray):
            T_mask = (Teff >= Trange[0]) & (Teff <= Trange[1])
        else:
            T_mask = Teff == Trange
            data = np.delete(data, 0, 1)
        mask = lg_mask & T_mask & MH_mask
    else:
        raise IOError("Unknown datafile structure or wrongly defined location.")

    data = data[mask, :]
    return data


def interpolated_LD_param(logg, Teff, MH, mTurb, logg_range=np.array([0, 7]), Trange=np.array([2000, 9000]),
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
    # # Flexibility depending on size of parameter space (how many have grid points) and table structure used
    if loc=='datafiles/tess_ldquad_table25.dat':
        if isinstance(logg_range, np.ndarray):
            eval_points = np.append(eval_points, logg)
        if isinstance(Trange, np.ndarray):
            eval_points = np.append(eval_points, Teff)
        if isinstance(MH_range, np.ndarray):
            eval_points = np.append(eval_points, MH)
        if isinstance(mTurb_range, np.ndarray):
            eval_points = np.append(eval_points, mTurb)
    elif loc == 'datafiles/kepler_sing_table.dat':
        if isinstance(Trange, np.ndarray):
            eval_points = np.append(eval_points, Teff)
        if isinstance(logg_range, np.ndarray):
            eval_points = np.append(eval_points, logg)
        if isinstance(MH_range, np.ndarray):
            eval_points = np.append(eval_points, MH)
    else:
        raise IOError("Unknown datafile structure or wrongly defined location.")

    eval_points = np.reshape(eval_points, (1, eval_points.size))
    res = interp.griddata(points, vals, eval_points, method='cubic')

    return res[0, :]


def save_LD_to_infile(LD_param_MS=None, LD_param_RG=None, loc_infile='jktebop/infile.TESS'):
    """
    Convenience function to use for updating limb-darkening parameters directly in the JKTEBOP input file.
    One or both components can be filled at a time.
    """
    if isinstance(LD_param_MS, np.ndarray) and isinstance(LD_param_RG, np.ndarray):
        LD_a = np.array([LD_param_MS[0], LD_param_RG[0]])
        LD_b = np.array([LD_param_MS[1], LD_param_RG[1]])
        with open(loc_infile, "r") as f:
            list_of_lines = f.readlines()
            list_of_lines[7] = " " + str(LD_a[0]) + " " + str(LD_a[1]) \
                               + "    LD star A (linear coeff)   LD star B (linear coeff)\n"
            list_of_lines[8] = " " + str(LD_b[0]) + " " + str(LD_b[1]) \
                               + "    LD star A (nonlin coeff)   LD star B (nonlin coeff)\n"
    elif isinstance(LD_param_MS, np.ndarray):
        with open(loc_infile, "r") as f:
            list_of_lines = f.readlines()
            list_of_lines[7] = " " + str(LD_param_MS[0]) + " " + list_of_lines[7].split()[1] \
                               + "    LD star A (linear coeff)   LD star B (linear coeff)\n"
            list_of_lines[8] = " " + str(LD_param_MS[1]) + " " + list_of_lines[8].split()[1] \
                               + "    LD star A (nonlin coeff)   LD star B (nonlin coeff)\n"
    elif isinstance(LD_param_RG, np.ndarray):
        with open(loc_infile, "r") as f:
            list_of_lines = f.readlines()
            list_of_lines[7] = " " + list_of_lines[7].split()[0] + " " + str(LD_param_RG[0])\
                               + "    LD star A (linear coeff)   LD star B (linear coeff)\n"
            list_of_lines[8] = " " + list_of_lines[8].split()[0] + " " + str(LD_param_RG[1]) \
                               + "    LD star A (nonlin coeff)   LD star B (nonlin coeff)\n"
    else:
        raise ValueError("No LD parameters to update were given.")

    with open(loc_infile, "w") as f:
        f.writelines(list_of_lines)


def jktebop_iterator(n_iter=4, loc_infile='jktebop_tess/infile.TESS', loc_jktebop='jktebop_tess/',
                     loc_ld_table='datafiles/tess_ldquad_table25.dat'):
    """
    Calls JKTEBOP to perform fit, extracts key parameters, calculates MS effective temperature,
    finds Limb-darkening parameters, and repeats iteratively.
    :param n_iter:      number of iterations needed
    :param loc_infile:  location of JKTEBOP input file
    :param loc_jktebop: location of JKTEBOP folder
    :param loc_ld_table: location of Limb darkening table
    """
    T_RG = 5042
    MH = -0.5
    mTurb = 2.0
    for i in range(0, n_iter+1):
        print("")
        print("Iteration ", i)
        subprocess.run("cd " + loc_jktebop + " && make clean -s && make -s", shell=True)
        _, jktebop_vals = read_jktebop_output(['Log surface gravity of star A (cgs):',
                                               'Log surface gravity of star B (cgs):', 'Radius of star A (Rsun)',
                                               'Radius of star B (Rsun)', 'Stellar light ratio (phase 0.1706):'],
                                               loc=loc_jktebop+'param.out')
        [L_ratio, R_MS, R_RG, loggMS, loggRG] = jktebop_vals
        print("Using T_RG=", T_RG, "  MH=", MH, "  mTurb=", mTurb)
        print("log g MS         ", loggMS)
        print("log g RG         ", loggRG)
        print("Radius MS        ", R_MS)
        print("Radius RG        ", R_RG)
        print("L_ratio          ", L_ratio)
        if loc_jktebop=='jktebop_tess/' or loc_jktebop=='jktebop_tess':
            spectral_response=tess_spectral_response
        elif loc_jktebop=='jktebop_kepler/' or loc_jktebop=='jktebop_kepler':
            spectral_response=kepler_spectral_response
        else:
            raise AttributeError("Unknown spectral response")
        T_MS = find_T2(R_RG, R_MS, T_RG, L_ratio, spectral_response)
        if i==0 or i==1:
            T_MS -= 500
        print("Calculated T_MS  ", T_MS)
        LD_param_MS = interpolated_LD_param(loggMS, T_MS, MH, mTurb, loc=loc_ld_table)
        LD_param_RG = interpolated_LD_param(loggRG, T_RG, MH, mTurb, loc=loc_ld_table)
        print("LD_param_MS      ", LD_param_MS)
        print("LD_param_RG      ", LD_param_RG)
        save_LD_to_infile(LD_param_MS, LD_param_RG, loc_infile=loc_infile)


jktebop_iterator(n_iter=4)


